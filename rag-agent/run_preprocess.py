import json
import os
from pathlib import Path
from src.preprocess import (
    clean_text,
    normalize_doc,
    extract_metadata,
    is_low_quality,
    detect_duplicates
)
from src.preprocess.handle_special_content import handle_special_content

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_files(directory: Path):
    docs = []
    for file_path in directory.glob("*"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append({
                    "id": file_path.name,
                    "text": f.read()
                })
    return docs


def save_docs(path: Path, docs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def main():
    raw_docs = load_raw_files(RAW_DIR)
    cleaned_docs = []

    for d in raw_docs:
        cleaned_text = clean_text(d["text"])

        # 특수 콘텐츠 처리
        cleaned_text = handle_special_content({"content": cleaned_text})["content"]

        meta = extract_metadata(d["text"])

        if is_low_quality(cleaned_text):
            continue

        normalized = normalize_doc(
            doc_id=d["id"],
            source="raw_source",
            content={
                "text": cleaned_text,
                "title": meta.get("title", d["id"]),
                "url": meta.get("url", ""),
                "raw_html": d["text"]  # 필요시 저장
            },
            metadata=meta
        )
        cleaned_docs.append(normalized)

    deduped_docs = detect_duplicates(cleaned_docs)
    save_docs(PROCESSED_DIR / "processed.json", deduped_docs)
    print(f"[INFO] 총 {len(deduped_docs)}개의 문서 저장 완료")


if __name__ == "__main__":
    main()
