import json
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os # 👈 [수정] os 라이브러리 추가

# --- 설정 부분 ---
# 3단계에서 생성된 입력 파일
INPUT_FILENAME = "processed_chunks.json"
# Elasticsearch 인덱스 이름
INDEX_NAME = "customs-docs-v1"
# 사용할 한국어 임베딩 모델
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask' # 👈 [확인] 이 부분도 multitask (t)로 되어있는지 확인!

# 👈 [수정] 로컬 실행 시 localhost, Docker 내부에서는 환경변수로 설정
ELASTICSEARCH_HOST = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
# -----------------

# ------------------- 메인 실행 부분 -------------------
if __name__ == "__main__":
    # 1. JSON 파일에서 데이터 로드
    print(f">> '{INPUT_FILENAME}' 파일에서 데이터를 로드합니다...")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다.")
        exit(1)
    
    # 2. 임베딩 모델 로드
    print(f">> '{EMBEDDING_MODEL}' 임베딩 모델을 로드합니다...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 임베딩 모델 로드 완료!")
    except Exception as e:
        print(f"❌ 오류: 임베딩 모델 로드에 실패했습니다. - {e}")
        exit(1)

    # 3. Elasticsearch 클라이언트 연결
    print(">> Elasticsearch에 연결합니다...")
    try:
        es_client = Elasticsearch(ELASTICSEARCH_HOST, verify_certs=False)
        
        # 연결 테스트
        if not es_client.ping():
            raise ConnectionError("Elasticsearch에 연결할 수 없습니다.")
        
        print("✅ Elasticsearch 연결 성공!")
    except Exception as e:
        print(f"❌ 오류: Elasticsearch 연결에 실패했습니다. - {e}")
        exit(1)

    # 4. Elasticsearch 인덱스 생성
    print(f">> '{INDEX_NAME}' 인덱스를 생성합니다...")
    
    index_mapping = {
        "properties": {
            "source": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "standard"},
            "content_vector": {
                "type": "dense_vector",
                "dims": model.get_sentence_embedding_dimension(),
                "index": True,
                "similarity": "cosine"
            }
        }
    }
    
    if es_client.indices.exists(index=INDEX_NAME):
        es_client.indices.delete(index=INDEX_NAME)
        print(f"   - 기존 '{INDEX_NAME}' 인덱스를 삭제했습니다.")
        
    es_client.indices.create(index=INDEX_NAME, mappings=index_mapping)
    print(f"✅ '{INDEX_NAME}' 인덱스 생성 완료!")

    # 5. 데이터 색인 (Bulk 작업)
    print(">> 데이터 임베딩 및 색인을 시작합니다...")
    
    actions = []
    for doc in tqdm(chunks_data, desc="   - 임베딩 및 색인 작업 진행 중"):
        try:
            embedding = model.encode(doc['content']).tolist()
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "source": doc['source'],
                    "content": doc['content'],
                    "content_vector": embedding
                }
            }
            actions.append(action)
        except Exception as e:
            print(f"⚠️  문서 처리 중 오류: {doc['source']} - {e}")

    try:
        bulk(es_client, actions)
        print("\n✅ Bulk 색인 작업 완료!")
    except Exception as e:
        print(f"\n❌ 오류: Bulk 색인 중 오류가 발생했습니다. - {e}")

    print(f"\n>> 모든 작업 완료! 총 {len(actions)}개의 문서가 '{INDEX_NAME}' 인덱스에 성공적으로 저장되었습니다.")