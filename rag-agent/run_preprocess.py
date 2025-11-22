import json
import os
from src.preprocess import (
    clean_text,
    normalize_doc,
    extract_metadata,
    is_low_quality,
    detect_duplicates,
    remove_boilerplate,
    extract_main_paragraphs,
)

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_files(directory):
    docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"id": filename, "text": f.read()})
    return docs


def save_docs(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def main():
    raw_docs = load_raw_files(RAW_DIR)

    cleaned_docs = []
    for d in raw_docs:
        text_cleaned = clean_text(d["text"])
        text_cleaned = remove_boilerplate(text_cleaned)
        text_main = extract_main_paragraphs(text_cleaned)

        meta = extract_metadata(d["text"])

        if is_low_quality(text_main):
            continue

        normalized = normalize_doc(
            doc_id=d["id"],
            source="raw_source",
            content=text_main,
            metadata=meta,
        )
        cleaned_docs.append(normalized)

    deduped_docs = detect_duplicates(cleaned_docs)

    save_docs(os.path.join(PROCESSED_DIR, "processed.json"), deduped_docs)


if __name__ == "__main__":
    main()