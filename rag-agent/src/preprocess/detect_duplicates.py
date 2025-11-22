import hashlib


def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_duplicates(docs: list) -> list:
    seen = set()
    unique_docs = []

    for doc in docs:
        h = get_text_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    return unique_docs
