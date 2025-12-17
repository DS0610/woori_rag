# 🏛️ CAG + RAG 관세청 AI 챗봇

캐시 증강 생성(CAG)과 검색 증강 생성(RAG)을 결합한 **관세청 전문 AI 챗봇**입니다.


## 프로젝트 프론트 화면
![프로젝트 스크린샷](https://github.com/user-attachments/assets/7633452c-213b-4fc0-9aeb-1bd2add13c7d)
![CAG HIT 예시](https://github.com/user-attachments/assets/788b607c-13ca-4fcf-91af-9257408cdf5f)

## 📋 프로젝트 개요

| 기능 | 설명 |
|------|------|
| ⚡ **CAG** | Redis 벡터 캐시로 FAQ 즉시 응답 (0.1~0.3초) |
| 📚 **RAG** | Elasticsearch 문서 검색 + LLM 답변 생성 (30~60초) |
| 💬 **Streamlit UI** | 대화형 채팅 인터페이스 |

### 워크플로우

```
사용자 질문
    ↓
CAG Cache 조회 (similarity ≥ 0.85)
    ├─ HIT → ⚡ 즉시 응답
    └─ MISS → RAG 파이프라인
                 ├─ Elasticsearch 문서 검색
                 ├─ LLM 답변 생성 (llama3.2:3b)
                 └─ Dynamic Cache 저장
```

---

## 🔧 기술 스택

| 구성 요소 | 기술 |
|----------|------|
| 벡터 캐시 | Redis Stack |
| 문서 검색 | Elasticsearch 8.x |
| 임베딩 | jhgan/ko-sroberta-multitask |
| LLM | Ollama + llama3.2:3b |
| UI | Streamlit |

---

## 📁 프로젝트 구조

```
rag_project/
├── app/
│   ├── cag.py              # CAG Cache 모듈
│   ├── cag_rag_chain.py    # CAG→RAG 통합 체인
│   └── streamlit_app.py    # Streamlit UI
├── rag/
│   ├── app/
│   │   ├── datacollect.py      # 웹 크롤링 + PDF 추출
│   │   ├── preprocess_data.py  # 텍스트 청킹
│   │   └── index_data.py       # ES 인덱싱
│   └── pdf_files/              # RAG용 PDF
├── data/
│   └── 2024 관세행정 민원상담 사례집.pdf  # CAG용 FAQ
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 실행 방법

### 1. Docker 서비스 시작
```bash
docker-compose up -d
```

### 2. Ollama 모델 다운로드
```bash
docker exec -it my-ollama ollama pull llama3.2:3b
```

### 3. Python 의존성 설치
```bash
pip install -r requirements.txt
pip install langchain-text-splitters pymupdf
```

### 4. RAG 데이터 준비 (Elasticsearch)
```bash
cd rag
python app/datacollect.py      # 데이터 수집
python app/preprocess_data.py  # 청킹
python app/index_data.py       # ES 인덱싱
```

### 5. CAG 데이터 준비 (Redis)
```bash
cd /path/to/rag_project
python -c "
from app.cag import CAGCache
cag = CAGCache(force_recreate_index=True)
cag.pre_cache_pdf('./data/2024 관세행정 민원상담 사례집.pdf')
"
```

### 6. Streamlit 앱 실행
```bash
streamlit run app/streamlit_app.py
```

브라우저: `http://localhost:8501`

---

## 🧪 테스트 질문

| 유형 | 질문 예시 | 응답 시간 |
|------|----------|----------|
| ⚡ CAG | "관세 납부 방법을 알려주세요" | 0.1~0.3초 |
| 📚 RAG | "여행자 휴대품 면세 한도는?" | 30~60초 |
| ❌ 불가 | "오늘 날씨 어때?" | - |

---

## 📊 데이터 소스

| 저장소 | 데이터 | 문서 수 |
|-------|--------|--------|
| Redis (CAG) | 민원상담 사례집 PDF | 1,337 Q&A |
| Elasticsearch (RAG) | 관세청 웹 + PDF | 77 청크 |

---

## ⚙️ 설정값

| 설정 | 값 |
|------|-----|
| CAG 임계값 | 0.85 |
| LLM 타임아웃 | 120초 |
| LLM 모델 | llama3.2:3b |

---

## 🔧 트러블슈팅

### Redis 캐시 초기화
```bash
docker exec -it my-redis redis-cli FLUSHALL
# 이후 CAG Pre-Cache 재로딩 필요
```

### Kibana (ES UI) 접속
```
http://localhost:5601
```

---

## 📝 라이선스

MIT License