import json
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

DATA_DIR = "./data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings", "embeddings.json")
VECTORSTORE_FILE = os.path.join(DATA_DIR, "vectorstore", "faiss_index")

os.makedirs(os.path.dirname(VECTORSTORE_FILE), exist_ok=True)

def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        docs.append(Document(page_content=item["text"], metadata=item.get("metadata", {})))
    return docs

def main():
    print("[INFO] Loading embeddings...")
    docs = load_embeddings(EMBEDDINGS_FILE)

    print("[INFO] Creating FAISS vector store with HuggingFace embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    print(f"[INFO] Saving vector store to {VECTORSTORE_FILE}...")
    vectorstore.save_local(VECTORSTORE_FILE)
    print("[INFO] Vector store creation finished.")

if __name__ == "__main__":
    main()
