import json
import os
import matplotlib.pyplot as plt

BASE_DIR = "./data/chunks_test"
CHUNK_SIZES = [500, 1000, 1500]
SAVE_DIR = "./figures"

os.makedirs(SAVE_DIR, exist_ok=True)

num_chunks = []
avg_lengths = []

for size in CHUNK_SIZES:
    path = os.path.join(BASE_DIR, f"chunked_{size}.json")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 1. chunk 개수
    num_chunks.append(len(chunks))

    # 2. 평균 chunk 길이 (단어 기준)
    lengths = [len(c["text"].split()) for c in chunks]
    avg_lengths.append(sum(lengths) / len(lengths))

# -----------------------------
# 그래프 1: Chunk 개수 비교
# -----------------------------
plt.figure()
plt.bar(CHUNK_SIZES, num_chunks, width=200)
plt.xlabel("Chunk Size")
plt.ylabel("Number of Chunks")
plt.title("Number of Chunks vs Chunk Size")
plt.savefig(
    os.path.join(SAVE_DIR, "num_chunks_vs_chunk_size.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------
# 그래프 2: 평균 Chunk 길이
# -----------------------------
plt.figure()
plt.plot(CHUNK_SIZES, avg_lengths, marker="o")
plt.xlabel("Chunk Size")
plt.ylabel("Average Chunk Length (words)")
plt.title("Average Chunk Length vs Chunk Size")
plt.savefig(
    os.path.join(SAVE_DIR, "avg_chunk_length_vs_chunk_size.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# -----------------------------
# 설정
# -----------------------------
BASE_DIR = "./data/chunks_test"
CHUNK_SIZES = [500, 1000, 1500]
TOP_K = 5
SAVE_DIR = "./figures"

os.makedirs(SAVE_DIR, exist_ok=True)

# 예시 query (여러 개면 리스트로)
QUERIES = [
    "RAG 시스템에서 chunk size의 역할은 무엇인가?",
    "chunk overlap이 검색 성능에 미치는 영향은?"
]

# -----------------------------
# 모델 로드
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

avg_similarities = []

for size in CHUNK_SIZES:
    path = os.path.join(BASE_DIR, f"chunked_{size}.json")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    # chunk 임베딩
    chunk_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    per_query_scores = []

    for q in QUERIES:
        q_emb = model.encode(q)

        sims = [
            cosine_similarity(q_emb, emb)
            for emb in chunk_embeddings
        ]

        topk_avg = np.mean(sorted(sims, reverse=True)[:TOP_K])
        per_query_scores.append(topk_avg)

    avg_similarities.append(np.mean(per_query_scores))


x = np.arange(len(CHUNK_SIZES))

plt.figure()
plt.bar(x, avg_similarities, width=0.6)
plt.xticks(x, CHUNK_SIZES)
plt.xlabel("Chunk Size")
plt.ylabel("Average Cosine Similarity (Top-k)")
plt.title("Average Cosine Similarity vs Chunk Size")

plt.savefig(
    os.path.join(SAVE_DIR, "cosine_similarity_vs_chunk_size.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
