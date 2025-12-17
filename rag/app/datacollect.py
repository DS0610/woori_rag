import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import os
import json 

# --- 설정 부분 ---
# PDF 파일이 저장된 폴더 이름
PDF_FOLDER = "pdf_files"

# 처리할 데이터 목록: 사용자님의 파일명에 맞게 수정되었습니다.
DATA_SOURCES = [
    # --- 웹에서 직접 가져올 HTML 페이지 ---
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2835&cntntsId=827"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2836&cntntsId=828"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2838&cntntsId=830"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2837&cntntsId=829"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=8451&cntntsId=2800"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=8457&cntntsId=2807"},
    {'type': 'html', 'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2808&cntntsId=2808"},

    # --- 로컬 폴더에서 읽어올 PDF 파일 ---
    {'type': 'pdf',  'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=2837&cntntsId=829", 'path': "250321_한 여행자휴대품통관 길라잡이.pdf"},
    {'type': 'pdf',  'source': "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?mi=8457&cntntsId=2800", 'path': "해외 이사물품 통관 안내서(2024년).pdf"},
]
# -----------------

def fetch_data_from_sources(sources):
    all_docs = []
    for item in sources:
        source_url = item['source']
        try:
            if item['type'] == 'pdf':
                file_path = os.path.join(PDF_FOLDER, item['path'])
                if not os.path.exists(file_path):
                    print(f"⚠️  파일을 찾을 수 없음: {file_path}")
                    continue
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                all_docs.append({'source': source_url, 'content': text})
                print(f"✅ 로컬 PDF 처리 완료: {file_path}")
            elif item['type'] == 'html':
                response = requests.get(source_url, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = soup.select_one('#cntntsView') or soup.select_one('div.co-body')
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                    all_docs.append({'source': source_url, 'content': text})
                    print(f"✅ HTML 페이지 처리 완료: {source_url}")
                else:
                    print(f"⚠️  HTML 내용을 찾을 수 없음: {source_url}")
        except Exception as e:
            print(f"❌ 오류 발생: {source_url} - {e}")
    return all_docs

# --- 실행 부분 ---
if __name__ == "__main__":
    print(">> 데이터 수집을 시작합니다...")
    crawled_data = fetch_data_from_sources(DATA_SOURCES)
    print("\n>> 데이터 수집이 완료되었습니다.")
    
    # --- ⭐⭐⭐ 데이터를 JSON 파일로 저장하는 부분 ⭐⭐⭐ ---
    if crawled_data:
        file_name = "crawled_data.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(crawled_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 데이터가 '{file_name}' 파일로 성공적으로 저장되었습니다.")
    # ----------------------------------------------------

    print("="*40)
    print(f"총 {len(crawled_data)}개의 문서를 수집했습니다.")
    print("="*40 + "\n")

    # (화면 출력 부분은 그대로 유지)
    if crawled_data:
        for i, doc in enumerate(crawled_data):
            print(f"--- 문서 {i+1} ---")
            print(f"출처 (원본 URL): {doc['source']}")
            print(f"내용 (앞 150자): {doc['content'][:150]}...\n")