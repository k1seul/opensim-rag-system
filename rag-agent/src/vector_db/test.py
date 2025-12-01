from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

VECTORSTORE_FILE = "./data/vectorstore/faiss_index"

# 1. 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. 기존 FAISS 벡터스토어 로드
vectorstore_faiss = FAISS.load_local(VECTORSTORE_FILE, embedding_model, allow_dangerous_deserialization=True)

# 3. 쿼리 테스트
query_faiss = "How do I set up a musculoskeletal model in OpenSim?"
retrieved_docs_faiss = vectorstore_faiss.similarity_search(query_faiss, k=2)

print(f"\n--- Query: '{query_faiss}' ---")
print(f"--- Found {len(retrieved_docs_faiss)} relevant documents ---")

for i, doc in enumerate(retrieved_docs_faiss, 1):
    print(f"\n--- [Result {i}] ---")
    print(doc.page_content)
    print(doc.metadata)
