# streamlit_app.py
"""
CAG + RAG Streamlit Chatbot UI
- 캐시 HIT/MISS 상태 표시
- 대화 히스토리 관리
"""

import streamlit as st
import time
import sys
import os

# 상위 디렉토리 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cag_rag_chain import get_chain

# 페이지 설정
st.set_page_config(
    page_title="관세청 AI 챗봇",
    page_icon="🏛️",
    layout="centered",
)

# 스타일
st.markdown("""
<style>
    .cache-hit {
        background-color: #d4edda;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        color: #155724;
    }
    .cache-miss-rag {
        background-color: #cce5ff;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        color: #004085;
    }
    .cache-miss-none {
        background-color: #f8d7da;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        color: #721c24;
    }
    .stChatMessage {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.title("🏛️ 관세청 AI 챗봇")
st.caption("관세, 통관, 해외직구 등에 대해 질문해주세요")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("🔧 시스템 초기화 중..."):
        try:
            st.session_state.chain = get_chain()
            st.success("✅ 시스템 준비 완료!")
        except Exception as e:
            st.error(f"❌ 초기화 실패: {e}")
            st.info("Redis, Elasticsearch, Ollama가 실행 중인지 확인하세요.")
            st.stop()


def get_status_badge(source: str, cache_hit: bool) -> str:
    """상태 배지 HTML 반환"""
    if cache_hit:
        return '<span class="cache-hit">⚡ CAG HIT (캐시)</span>'
    elif source == "RAG":
        return '<span class="cache-miss-rag">📚 RAG (문서 검색)</span>'
    else:
        return '<span class="cache-miss-none">❌ 검색 불가</span>'


# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "status" in message:
            st.markdown(message["status"], unsafe_allow_html=True)

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("🤔 답변 생성 중..."):
            start_time = time.time()
            
            # CAG → RAG 체인 호출
            result = st.session_state.chain.invoke({"question": prompt})
            
            elapsed_time = time.time() - start_time

        # 답변 표시
        st.markdown(result["answer"])
        
        # 상태 배지 표시
        status_badge = get_status_badge(result["source"], result["cache_hit"])
        st.markdown(
            f'{status_badge} <small style="color: gray;">({elapsed_time:.2f}초)</small>',
            unsafe_allow_html=True,
        )

    # 히스토리에 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "status": f'{status_badge} <small style="color: gray;">({elapsed_time:.2f}초)</small>',
    })

# 사이드바
with st.sidebar:
    st.header("ℹ️ 정보")
    st.markdown("""
    **워크플로우:**
    1. ⚡ **CAG HIT**: 캐시에서 즉시 답변
    2. 📚 **RAG**: 문서 검색 후 답변 생성
    3. ❌ **검색 불가**: 관련 문서 없음
    """)
    
    st.divider()
    
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.caption("Redis + Elasticsearch + Ollama")
