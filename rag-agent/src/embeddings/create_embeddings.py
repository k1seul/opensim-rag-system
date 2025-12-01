import json
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

CHUNK_FILE = Path("data/chunks/chunked.json")
OUTPUT_FILE = Path("data/embeddings/embeddings.json")

def main():
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))

    # 임베딩 모델 초기화
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    embedding_data = []
    texts = [c["text"] for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    for chunk, vector in zip(chunks, vectors):
        embedding_data.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "url": chunk["url"],
            "title": chunk["title"],
            "text": chunk["text"],
            "embedding": vector
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(embedding_data)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
