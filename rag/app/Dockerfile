# 1. 베이스 이미지 선택
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 모든 소스 코드 복사
COPY . .

# 5. [수정] Windows(CRLF) 파일을 Linux(LF)로 변환
#    dos2unix 도구를 설치하고, run.sh 파일에 적용합니다.
RUN apt-get update && apt-get install -y dos2unix \
    && dos2unix ./run.sh

# 6. run.sh 파일에 실행 권한 부여
RUN chmod +x run.sh

# 7. 컨테이너가 시작될 때 run.sh 스크립트 실행
CMD ["./run.sh"]