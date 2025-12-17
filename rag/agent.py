import json
import os 
import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import requests

# (프록시 우회 설정은 그대로 둡니다)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# --- 설정 (이하 동일) ---
ELASTICSEARCH_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "customs-docs-v1"
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask' 
OLLAMA_API_URL = "http://localhost:11434/api/chat" 
OLLAMA_MODEL = "llama3:8b" 

# --- ⚡️ [수정] AI 에이전트 시스템 프롬프트 (한국어 지시 강화) ⚡️ ---
SYSTEM_PROMPT = """
당신은 관세청의 공식 AI 에이전트 '커스텀-봇'입니다.
당신의 임무는 오직 제공되는 [관세청 공식 자료]를 근거로 하여 사용자의 질문에 답변하는 것입니다.

[지시 사항]
1. 사용자의 [질문]에 답변하기 위해, [관세청 공식 자료]에서만 근거를 찾으세요.
2. 답변은 명확하고, 이해하기 쉬운 한국어로 친절하게 제공해야 합니다.
3. 만약 [관세청 공식 자료]에 답변의 근거가 되는 내용이 없다면, "죄송합니다만, 제공된 자료에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.
4. 절대 [관세청 공식 자료]에 없는 내용을 추측하거나 임의의 정보를 생성하지 마세요.
5. [매우 중요] 모든 답변은 반드시, 무조건, 예외 없이 **한국어로만** 작성해야 합니다. (Do not write in English.)
"""
# ----------------------------------------------------------------

USER_PROMPT_TEMPLATE = """
[관세청 공식 자료]
{retrieved_documents}
---
[질문]
{user_query}
"""

class CustomsRAGAgent:
    def __init__(self):
        # (이하 __init__ 함수 내용은 동일)
        print("AI 에이전트 초기화를 시작합니다...")
        try:
            print(f"'{EMBEDDING_MODEL}' 임베딩 모델을 로드합니다...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("✅ 임베딩 모델 로드 완료.")
            print(f"'{ELASTICSEARCH_HOST}'에 연결을 시도합니다...")
            self.es_client = Elasticsearch(ELASTICSEARCH_HOST)
            if not self.es_client.ping():
                raise ConnectionError("Elasticsearch에 연결할 수 없습니다.")
            print("✅ Elasticsearch 연결 성공.")
        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {e}")
            print("Elasticsearch Docker 컨테이너가 실행 중인지, 모델 이름이 정확한지 확인해주세요.")
            exit()

    def retrieve(self, query, top_k=3):
        # (이하 retrieve 함수 내용은 동일)
        print(f"\n[1/3] '{query}' (와)과 관련된 문서를 검색합니다...")
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            knn_query = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 10
            }
            response = self.es_client.search(
                index=INDEX_NAME,
                knn=knn_query,
                source=["source", "content"],
                size=top_k
            )
            hits = response['hits']['hits']
            if not hits:
                print("⚠️  검색된 문서가 없습니다.")
                return ""
            context = ""
            for i, hit in enumerate(hits):
                context += f"\n--- 문서 {i+1} (출처: {hit['_source']['source']}) ---\n"
                context += hit['_source']['content']
                context += "\n-----------------------------------\n"
            print(f"✅ 총 {len(hits)}개의 관련 문서를 찾았습니다.")
            return context
        except Exception as e:
            print(f"❌ 문서 검색 중 오류 발생: {e}")
            return ""

    def generate_answer(self, query, context):
        # (이하 generate_answer 함수 내용은 동일)
        print("[2/3] 검색된 문서를 바탕으로 답변을 생성합니다...")
        user_content = USER_PROMPT_TEMPLATE.format(
            retrieved_documents=context,
            user_query=query
        )
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "stream": False 
        }
        proxies_to_use = {
            "http": None,
            "https": None,
        }
        try:
            response = requests.post(
                OLLAMA_API_URL, 
                json=payload, 
                timeout=60,
                proxies=proxies_to_use 
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("message", {}).get("content", "").strip()
            print("✅ 답변 생성 완료.")
            return answer
        except requests.exceptions.ConnectionError:
            print(f"❌ API 호출 오류: Ollama 서버({OLLAMA_API_URL})에 연결할 수 없습니다.")
            print("Ollama가 로컬에서 실행 중인지 확인해주세요.")
        except requests.exceptions.HTTPError as e:
            print(f"❌ API 호출 오류: {e}") 
        except requests.exceptions.Timeout:
            print(f"❌ API 호출 오류: 응답 시간(60초)을 초과했습니다.")
        except Exception as e:
            print(f"❌ API 호출 중 알 수 없는 오류 발생: {e}")
        return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

    def ask(self, query):
        # (이하 ask 함수 내용은 동일)
        context_docs = self.retrieve(query)
        if not context_docs:
            return "죄송합니다. 질문에 대한 관련 문서를 찾을 수 없습니다."
        answer = self.generate_answer(query, context_docs)
        print("[3/3] 최종 답변을 반환합니다.")
        return answer

def main():
    # (이하 main 함수 내용은 동일)
    try:
        agent = CustomsRAGAgent()
    except Exception as e:
        return
    print("\n" + "="*50)
    print("관세 행정 AI 에이전트 '커스텀-봇'입니다.")
    print("개인 통관, 해외 직구 등에 대해 무엇이든 물어보세요.")
    print(f"(LLM: {OLLAMA_MODEL} @ Ollama)")
    print("(종료하시려면 'q' 또는 'exit'를 입력하세요.)")
    print("="*50)
    while True:
        query = input("\n[질문 입력] > ")
        if query.lower() in ['q', 'exit']:
            print("에이전트를 종료합니다.")
            break
        if not query:
            continue
        start_time = time.time()
        answer = agent.ask(query)
        end_time = time.time()
        print("\n[커스텀-봇 답변] 🤖:")
        print(answer)
        print(f"\n(답변 생성 시간: {end_time - start_time:.2f}초)")

if __name__ == "__main__":
    main()