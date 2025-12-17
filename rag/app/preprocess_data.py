import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 설정 부분 ---
# 1단계에서 생성된 입력 파일 이름
INPUT_FILENAME = "crawled_data.json"
# 2, 3단계 완료 후 저장할 출력 파일 이름
OUTPUT_FILENAME = "processed_chunks.json"
# -----------------

# ------------------- 2단계: 데이터 전처리 (Preprocessing) -------------------
def clean_text(text):
    """텍스트에서 불필요한 공백, 줄바꿈 등을 제거합니다."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\r\n]+', '\n', text)
    return text.strip()

# ------------------- 메인 실행 부분 -------------------
if __name__ == "__main__":
    # 1. JSON 파일 읽어오기
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)
        print(f"✅ '{INPUT_FILENAME}' 파일에서 {len(crawled_data)}개의 문서를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_FILENAME}' 파일을 찾을 수 없습니다.")
        print(">> 먼저 1단계 데이터 수집 코드를 실행하여 파일을 생성해주세요.")
        exit() # 파일이 없으면 프로그램 종료

    # 2. 텍스트 데이터 전처리 실행
    print("\n>> 2단계: 텍스트 데이터 전처리를 시작합니다...")
    for doc in crawled_data:
        # 'content' 키의 값을 전처리된 텍스트로 교체합니다.
        doc['content'] = clean_text(doc['content'])
    print("✅ 모든 문서 전처리 완료!")

    # 3. 데이터 분할(Chunking) 실행
    print("\n>> 3단계: 데이터 분할(Chunking)을 시작합니다...")
    
    # 3-1. 분할 기준 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    # 3-2. 모든 문서를 순회하며 청크로 분할
    all_chunks = []
    for doc in crawled_data:
        # 전처리된 내용을 기반으로 텍스트를 분할합니다.
        chunks = text_splitter.split_text(doc['content'])
        
        # 각 청크에 원본 URL 정보를 추가하여 저장합니다.
        for chunk_text in chunks:
            all_chunks.append({
                'source': doc['source'],
                'content': chunk_text
            })

    print(f"✅ 데이터 분할 완료! 총 {len(crawled_data)}개의 문서에서 {len(all_chunks)}개의 청크를 생성했습니다.")

    # 4. 최종 결과를 JSON 파일로 저장
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print(f"\n>> ✅ 모든 청크 데이터가 '{OUTPUT_FILENAME}' 파일로 성공적으로 저장되었습니다.")

    # 결과 확인: 생성된 청크 중 처음 2개만 출력
    print("\n--- 생성된 청크 예시 ---")
    for i, chunk in enumerate(all_chunks[:2]):
        print(f"\n[청크 {i+1}]")
        print(f"출처: {chunk['source']}")
        print(f"내용: {chunk['content']}")