# analyze_content.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import random
import os

# 파일 경로 설정
CONTENT_JSON = "data/processed/processed.json"
OUTPUT_DIR = "data/processed/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON 로드
with open(CONTENT_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

# 글자 수 / 단어 수 계산
lengths_char = [len(d['content']) for d in docs]
lengths_word = [len(re.findall(r'\w+', d['content'])) for d in docs]

print(f"Total documents: {len(docs)}")
print(f"Character count - min: {min(lengths_char)}, max: {max(lengths_char)}, mean: {sum(lengths_char)//len(lengths_char)}")
print(f"Word count - min: {min(lengths_word)}, max: {max(lengths_word)}, mean: {sum(lengths_word)//len(lengths_word)}")

# Random sample extraction and save
sample_docs = random.sample(docs, min(20, len(docs)))
with open(os.path.join(OUTPUT_DIR, "random_sample_docs.txt"), "w", encoding="utf-8") as f:
    for i, d in enumerate(sample_docs, 1):
        f.write(f"[{i}] doc_id: {d['doc_id']}\n")
        f.write(d['content'][:1000])
        f.write("\n---\n")
print(f"Random sample documents saved: {OUTPUT_DIR}/random_sample_docs.txt")

# Keyword frequency analysis
all_words = []
for d in docs:
    all_words.extend(re.findall(r'\w+', d['content']))

counter = Counter(all_words)
most_common_words = counter.most_common(30)

# Save top 30 words
with open(os.path.join(OUTPUT_DIR, "top_words.txt"), "w", encoding="utf-8") as f:
    for w, c in most_common_words:
        f.write(f"{w}: {c}\n")
print(f"Top 30 words saved: {OUTPUT_DIR}/top_words.txt")

# Visualization
sns.set(style="whitegrid")

# Character count distribution
plt.figure(figsize=(10, 6))
sns.histplot(lengths_char, bins=50, kde=True)
plt.title("Character Count Distribution")
plt.xlabel("Characters")
plt.ylabel("Number of Documents")
plt.savefig(f"{OUTPUT_DIR}/char_length_distribution.png")
plt.close()

# Word count distribution
plt.figure(figsize=(10, 6))
sns.histplot(lengths_word, bins=50, kde=True)
plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Number of Documents")
plt.savefig(f"{OUTPUT_DIR}/word_length_distribution.png")
plt.close()

# Top 30 word frequency bar plot
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts))
plt.title("Top 30 Word Frequencies")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.xlabel("Word")
plt.savefig(f"{OUTPUT_DIR}/top_words.png")
plt.close()

print(f"Analysis results and plots saved in {OUTPUT_DIR}.")
