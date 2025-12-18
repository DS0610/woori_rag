# cag_rag_chain.py
"""
CAG + RAG 통합 체인 모듈 (LangChain 기반)
- CAG HIT: 캐시된 답변 즉시 반환
- CAG MISS: RAG 파이프라인으로 문서 검색 후 답변 생성
"""

import os
import sys

# 상위 디렉토리 경로 추가 (rag 모듈 import용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
from app.cag import CAGCache

# Elasticsearch 기반 RAG 컴포넌트
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# LangChain 컴포넌트
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


class CAGRAGChain:
    """
    CAG → RAG Fallback 체인 (LangChain 기반)
    
    워크플로우:
    1. CAG 캐시 조회 (similarity >= threshold 시 HIT)
    2. MISS시 Elasticsearch 문서 검색
    3. 검색 결과 있으면 LLM으로 답변 생성
    4. Dynamic Cache에 저장
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        es_host: str = "http://localhost:9200",
        es_index: str = "customs-docs-v1",
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b",
        cache_threshold: float = 0.85,
    ):
        # CAG 캐시 초기화
        self.cag = CAGCache(
            redis_host=redis_host,
            redis_port=redis_port,
            force_recreate_index=False,
        )
        self.cache_threshold = cache_threshold

        # RAG 컴포넌트 초기화
        print("🔧 RAG 컴포넌트 초기화 중...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.es_client = Elasticsearch(es_host, verify_certs=False)
        self.es_index = es_index

        # Elasticsearch 연결 확인
        if not self.es_client.ping():
            print("⚠️ Elasticsearch 연결 실패 - RAG 기능이 제한됩니다")
        else:
            print("✅ Elasticsearch 연결 성공")

        # LangChain LLM 초기화
        print("🔧 LangChain LLM 초기화 중...")
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0,
            timeout=120,
        )
        print("✅ LangChain ChatOllama 연결 성공")

        # 시스템 프롬프트
        self.system_prompt = """당신은 관세청의 공식 AI 에이전트 '커스텀-봇'입니다.
당신의 임무는 오직 제공되는 [관세청 공식 자료]를 근거로 하여 사용자의 질문에 답변하는 것입니다.

[지시 사항]
1. 사용자의 [질문]에 답변하기 위해, [관세청 공식 자료]에서만 근거를 찾으세요.
2. 답변은 명확하고, 이해하기 쉬운 한국어로 친절하게 제공해야 합니다.
3. 만약 [관세청 공식 자료]에 답변의 근거가 되는 내용이 없다면, "죄송합니다만, 제공된 자료에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.
4. 절대 [관세청 공식 자료]에 없는 내용을 추측하거나 임의의 정보를 생성하지 마세요.
5. [매우 중요] 모든 답변은 반드시 **한국어로만** 작성해야 합니다.

[출력 형식]
- 마크다운(Markdown) 형식으로 예쁘게 답변하세요.
- 제목에는 ## 또는 ### 를 사용하세요.
- 중요한 내용은 **굵게** 표시하세요.
- 단계별 설명 시 1. 2. 3. 번호 목록을 사용하세요.
- 항목 나열 시 - 불릿포인트를 사용하세요.
- 금액이나 수치는 강조해서 표시하세요."""

        # LangChain 프롬프트 템플릿 구성
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("""[관세청 공식 자료]
{context}
---
[질문]
{question}""")
        ])

        # LangChain 체인 구성 (LCEL)
        self.rag_chain = self.prompt_template | self.llm | StrOutputParser()

    def _retrieve_documents(self, query: str, top_k: int = 3) -> str:
        """Elasticsearch에서 관련 문서 검색"""
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            knn_query = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 10,
            }
            response = self.es_client.search(
                index=self.es_index,
                knn=knn_query,
                source=["source", "content"],
                size=top_k,
            )
            hits = response["hits"]["hits"]
            if not hits:
                return ""

            context = ""
            for i, hit in enumerate(hits):
                context += f"\n--- 문서 {i+1} (출처: {hit['_source']['source']}) ---\n"
                context += hit["_source"]["content"]
                context += "\n-----------------------------------\n"
            return context
        except Exception as e:
            print(f"❌ 문서 검색 오류: {e}")
            return ""

    def _generate_answer(self, query: str, context: str) -> str:
        """LangChain LLM으로 답변 생성"""
        try:
            # LangChain LCEL 체인 실행
            answer = self.rag_chain.invoke({
                "context": context,
                "question": query
            })
            return answer.strip()
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg:
                return "❌ Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
            elif "timeout" in error_msg.lower():
                return "❌ 응답 시간을 초과했습니다. 잠시 후 다시 시도해주세요."
            else:
                return f"❌ 답변 생성 중 오류: {e}"

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, any]:
        """
        CAG → RAG 체인 실행
        
        Args:
            inputs: {"question": str}
        
        Returns:
            {"answer": str, "cache_hit": bool, "source": str}
        """
        question = inputs.get("question", "")
        if not question:
            return {"answer": "질문을 입력해주세요.", "cache_hit": False, "source": "NONE"}

        # 1. CAG 캐시 조회
        print(f"\n🔍 CAG 캐시 조회: {question[:30]}...")
        cached_answer = self.cag.check_cache(question, threshold=self.cache_threshold)

        if cached_answer:
            print("⚡ CAG HIT - 캐시된 답변 반환")
            return {"answer": cached_answer, "cache_hit": True, "source": "CAG"}

        # 2. RAG 파이프라인: 문서 검색
        print("📚 CAG MISS - RAG 파이프라인 실행")
        context = self._retrieve_documents(question)

        if not context:
            print("❌ 검색된 문서 없음")
            return {
                "answer": "죄송합니다. 질문에 대한 관련 문서를 검색할 수 없습니다.",
                "cache_hit": False,
                "source": "NONE",
            }

        # 3. LangChain LLM으로 답변 생성
        print("🤖 LangChain LLM 답변 생성 중...")
        answer = self._generate_answer(question, context)

        # 4. Dynamic Cache 저장 (에러 응답은 저장하지 않음)
        if not answer.startswith("❌"):
            self.cag.save_dynamic_cache(question, answer)
            print("💾 Dynamic Cache 저장 완료")
        else:
            print("⚠️ 에러 응답은 캐시에 저장하지 않음")

        return {"answer": answer, "cache_hit": False, "source": "RAG"}


# 싱글톤 인스턴스 (FastAPI, Streamlit에서 공유)
_chain_instance: Optional[CAGRAGChain] = None


def get_chain() -> CAGRAGChain:
    """CAGRAGChain 싱글톤 인스턴스 반환"""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = CAGRAGChain()
    return _chain_instance
