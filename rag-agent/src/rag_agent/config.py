from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

VECTORSTORE_DIR = "./data/vectorstore/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(VECTORSTORE_DIR, embedding_model, allow_dangerous_deserialization=True)
