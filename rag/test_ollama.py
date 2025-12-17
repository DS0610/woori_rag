import requests
import os

# --- ⚡️ 프록시 우회 설정 (모든 방법 동원) ⚡️ ---
# 1. 환경 변수 설정
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 2. requests 라이브러리에 명시적으로 프록시 없음을 선언
proxies_to_use = {
    "http": None,
    "https": None,
}
# ---------------------------------------------

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:8b"

print(f"Ollama 서버에 연결을 시도합니다: {OLLAMA_API_URL}")
print(f"프록시 설정: {proxies_to_use}")

payload = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "user", "content": "hi"}
    ],
    "stream": False
}

try:
    response = requests.post(
        OLLAMA_API_URL,
        json=payload,
        timeout=10,
        proxies=proxies_to_use # 👈 프록시 "없음"을 강제로 지정
    )

    # HTTP 상태 코드 확인
    print(f"\nHTTP 상태 코드: {response.status_code}")
    response.raise_for_status() # 4xx/5xx 에러 시 예외 발생

    # 성공 시 응답 출력
    result = response.json()
    print("--- ⭐️ 서버 응답 (성공) ⭐️ ---")
    print(result.get("message", {}).get("content", "No content found in message"))

except requests.exceptions.HTTPError as e:
    print("\n--- ❌ [HTTP 오류 발생] ❌ ---")
    print(f"오류: {e}")
    print("Ollama 서버가 4xx 또는 5xx 응답을 보냈습니다.")
    print("이것이 404라면, 프록시가 아닌 다른 문제입니다.")
    print(f"전체 응답 내용: {response.text}")

except requests.exceptions.ConnectionError as e:
    print("\n--- ❌ [연결 오류 발생] ❌ ---")
    print(f"오류: {e}")
    print("Ollama 서버가 꺼져있거나, 방화벽에 막혀있습니다.")

except Exception as e:
    print(f"\n--- ❌ [기타 알 수 없는 오류] ❌ ---")
    print(f"오류: {e}")