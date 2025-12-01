#!/bin/bash

# run_embeddings.sh
# Description: Create embeddings for text chunks using LangChain

# 환경 활성화 (필요시)
# source ~/miniconda3/bin/activate bkms

echo "[INFO] Start creating embeddings..."

python src/embeddings/create_embeddings.py

echo "[INFO] Embeddings creation finished."
