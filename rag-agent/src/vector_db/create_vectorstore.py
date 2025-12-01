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

    print("[INFO] Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 기존 vectorstore가 있으면 불러오기
    if os.path.exists(VECTORSTORE_FILE):
        print("[INFO] Loading existing FAISS vector store...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_FILE,
            embedding_model,
            allow_dangerous_deserialization=True  # <- 안전하므로 허용
        )
        print("[INFO] Adding new documents to existing vector store...")
        vectorstore.add_documents(docs)
    else:
        print("[INFO] Creating new FAISS vector store...")
        vectorstore = FAISS.from_documents(docs, embedding_model)

    print(f"[INFO] Saving vector store to {VECTORSTORE_FILE}...")
    vectorstore.save_local(VECTORSTORE_FILE)
    print("[INFO] Vector store creation finished.")

if __name__ == "__main__":
    main()
