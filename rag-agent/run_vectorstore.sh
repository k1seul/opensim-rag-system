#!/bin/bash
# FAISS vector store 생성

echo "[INFO] Start creating vector store..."
python src/vector_db/create_vectorstore.py
echo "[INFO] Vector store creation finished."
