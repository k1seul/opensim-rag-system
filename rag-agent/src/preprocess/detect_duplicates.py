import hashlib

def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def detect_duplicates(docs: list) -> list:
    seen = set()
    unique_docs = []

    for doc in docs:
        # content가 dict이면 text 필드만 사용
        content_text = doc["content"]["text"] if isinstance(doc["content"], dict) else str(doc["content"])
        h = get_text_hash(content_text)
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    return unique_docs
