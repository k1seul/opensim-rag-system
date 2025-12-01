# src/chunking/recursive_chunker.py
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_file(self, file_path: Path):
        """
        JSON 파일을 읽어 각 문서의 본문(text)만 추출하고, 
        RecursiveCharacterTextSplitter로 chunking 후 리스트 반환
        """
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to load JSON: {file_path}")
            return []

        chunks = []

        for item in data:
            # content가 str이면 JSON으로 변환, dict면 그대로 사용
            content_raw = item.get("content", {})
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            elif isinstance(content_raw, dict):
                content = content_raw
            else:
                continue

            # 본문 text만 추출
            text = content.get("text", "")
            if not text:
                continue

            split_texts = self.splitter.split_text(text)
            doc_id = item.get("doc_id", file_path.stem)

            for i, chunk in enumerate(split_texts):
                chunks.append({
                    "id": f"{doc_id}_{i}",
                    "source": str(file_path),
                    "url": content.get("url", ""),
                    "title": content.get("title", ""),
                    "text": chunk
                })

        return chunks

    def save_chunks(self, chunks, output_path: Path):
        """
        chunk 리스트를 JSON 파일로 저장
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
