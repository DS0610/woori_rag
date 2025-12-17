#!/bin/sh

# 스크립트 실행 중 하나라도 오류가 나면 즉시 중단
set -e

echo "--- 1. 데이터 수집 시작 (datacollect.py) ---"
python datacollect.py

echo "--- 2. 데이터 전처리 및 분할 시작 (preprocess_data.py) ---"
python preprocess_data.py

echo "--- 3. 데이터 색인 시작 (index_data.py) ---"
python index_data.py

echo "--- 모든 Docker 작업 완료 ---"