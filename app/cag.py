# cag_cache.py
import re
import json
from collections import deque
from typing import List, Dict, Optional

import redis
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class CAGCache:
    """
    ✅ CAG(캐시 증강) 전용 모듈
    - Redis + SentenceTransformer 기반 캐시 인덱스
    - PDF → Pre-Cache 적재
    - 캐시 조회(check_cache) + Dynamic Cache 저장
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_index: str = "cache_index",
        model_name: str = "jhgan/ko-sroberta-multitask",
        dynamic_cache_size: int = 5,
        force_recreate_index: bool = False,
    ):
        # 1) Redis & 모델 초기화
        self.r = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False  # 임베딩을 bytes로 저장해야 해서 False 유지
        )
        self.model = SentenceTransformer(model_name)
        self.CACHE_INDEX = cache_index
        self.user_cache = deque(maxlen=dynamic_cache_size)

        # 2) 인덱스 생성
        self._init_cache_index(force_recreate=force_recreate_index)

    # ------------------------------------------------------------
    # 임베딩 함수 (정규화 비활성화: Redis COSINE과 호환)
    # ------------------------------------------------------------
    def _embed(self, text: str) -> bytes:
        emb = self.model.encode(text, normalize_embeddings=False)
        return np.array(emb, dtype=np.float32).tobytes()

    # ------------------------------------------------------------
    # 캐시 인덱스 초기화
    # ------------------------------------------------------------
    def _init_cache_index(self, force_recreate: bool = True):
        if force_recreate:
            try:
                self.r.ft(self.CACHE_INDEX).dropindex(delete_documents=True)
                print(f"🗑️ 기존 {self.CACHE_INDEX} 인덱스 삭제 완료")
            except Exception:
                pass

        try:
            self.r.ft(self.CACHE_INDEX).info()
            print(f"ℹ️ {self.CACHE_INDEX} 이미 존재 (재사용)")
        except Exception:
            dim = len(self.model.encode("차원 확인", normalize_embeddings=False))
            self.r.ft(self.CACHE_INDEX).create_index(
                fields=[
                    VectorField("embedding", "FLAT", {
                        "TYPE": "FLOAT32",
                        "DIM": dim,
                        "DISTANCE_METRIC": "COSINE",
                    }),
                    TextField("text"),
                    TextField("source"),
                ],
                definition=IndexDefinition(
                    prefix=["cache:"],
                    index_type=IndexType.HASH,
                ),
            )
            print(f"✅ {self.CACHE_INDEX} 인덱스 생성 완료")

    # ------------------------------------------------------------
    # PDF Q–A 파싱 (본문 + 표 포함)
    # ------------------------------------------------------------
    def extract_qa_pairs(self, pdf_path: str) -> List[Dict[str, str]]:
        qa_pairs: List[Dict[str, str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                lines = text.split("\n") if text else []

                # 표 추출
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    rows = [
                        " | ".join([cell if cell else "" for cell in row])
                        for row in table
                    ]
                    table_texts.append("\n".join(rows))
                table_text_block = (
                    "\n\n[표 데이터]\n" + "\n\n".join(table_texts)
                    if table_texts
                    else ""
                )

                merged_text = text + table_text_block

                current_q: Optional[str] = None
                current_a: List[str] = []

                for line in merged_text.split("\n"):
                    line = line.strip()
                    # 질문 패턴 (네가 쓰던 정규식 그대로 사용)
                    if re.match(
                        r".*(\?|？|궁금합니다\.?|알려주세요\.?|무엇인가요\.?|어떻게.*|대해\s*설명.*|요약.*|문의(?:합니다|드립니다)\.?|설명(?:해\s*주|하여\s*주|바랍니다)\.?|알고\s*싶.*|요청(?:합니다|드립니다)\.?|유의사항$|절차$|방법$|기준$|대상$|요건$|처리$|신고$|수입$|수출$|반입$|검사$|허가$|확인$|통관$)$",
                        line,
                    ):
                        if current_q and current_a:
                            qa_pairs.append(
                                {
                                    "question": current_q,
                                    "answer": "\n".join(current_a).strip(),
                                }
                            )
                        current_q = line
                        current_a = []
                    elif current_q:
                        current_a.append(line)

                if current_q and current_a:
                    qa_pairs.append(
                        {
                            "question": current_q,
                            "answer": "\n".join(current_a).strip(),
                        }
                    )

        return qa_pairs

    # ------------------------------------------------------------
    # Pre-Cache (PDF → Redis 저장)
    # ------------------------------------------------------------
    def pre_cache_pdf(self, pdf_path: str):
        qa_list = self.extract_qa_pairs(pdf_path)
        print(f"📘 PDF에서 {len(qa_list)}개의 QA 추출 완료")

        for i, qa in enumerate(qa_list):
            key = f"cache:pdf:{i}"
            self.r.hset(
                key,
                mapping={
                    "embedding": self._embed(qa["question"]),
                    "text": qa["answer"],
                    "source": "pdf_pre_cache",
                },
            )
        print(f"💾 Redis에 {len(qa_list)}개 Pre-Cache 저장 완료")

    # ------------------------------------------------------------
    # 캐시 검색 (CAG)
    # ------------------------------------------------------------
    def check_cache(
        self,
        user_query: str,
        k: int = 3,
        threshold: float = 0.7,
    ) -> Optional[str]:
        """
        - Redis 벡터 검색으로 유사 질문 찾기
        - sim >= threshold 이면 캐시 HIT, 아니면 MISS
        """
        q_emb = self._embed(user_query)
        q = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .return_fields("text", "source", "score")
            .sort_by("score")
            .dialect(2)
        )

        try:
            res = self.r.ft(self.CACHE_INDEX).search(q, query_params={"vec": q_emb})
        except Exception as e:
            print("❌ 캐시 검색 오류:", e)
            return None

        if not res.docs:
            print("❌ 캐시에서 유사 문서 없음")
            return None

        # Redis KNN 검색의 score는 거리이므로, 1 - distance로 유사도 추정
        sim = 1 - float(res.docs[0].score)
        print(f"📊 유사도 점수: {sim:.2f}")
        if sim >= threshold:
            print(
                f"⚡ 캐시 HIT (유사도 {sim:.2f}) [source={res.docs[0].source}]"
            )
            return res.docs[0].text

        print(f"❌ 캐시 MISS (유사도 {sim:.2f} < {threshold})")
        return None

    # ------------------------------------------------------------
    # Dynamic Cache 저장 (최근 N개 유지)
    # ------------------------------------------------------------
    def save_dynamic_cache(self, query: str, answer: str):
        key = f"cache:dyn:{abs(hash(query)) % (10**8)}"
        self.r.hset(
            key,
            mapping={
                "embedding": self._embed(query),
                "text": answer,
                "source": "dynamic_cache",
            },
        )
        self.user_cache.append(key)
        if len(self.user_cache) > self.user_cache.maxlen:
            oldest = self.user_cache.popleft()
            self.r.delete(oldest)
            print(f"🗑️ 오래된 캐시 삭제: {oldest}")

        print(f"💾 Dynamic Cache 저장: {query[:30]}...")
