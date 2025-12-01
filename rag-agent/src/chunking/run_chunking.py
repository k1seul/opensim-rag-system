from pathlib import Path
from .recursive_chunker import RecursiveChunker

def run():
    processed_dir = Path("data/processed")
    chunk_dir = Path("data/chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=150)
    all_chunks = []

    # processed 디렉토리 내 모든 JSON 파일 처리
    for file_path in processed_dir.glob("*.json"):
        print(f"[INFO] Processing {file_path.name} ...")
        chunks = chunker.chunk_file(file_path)
        print(f"[INFO] {len(chunks)}개의 chunk 생성 완료")
        all_chunks.extend(chunks)

    # 모든 chunk를 하나의 JSON 파일로 저장
    output_file = chunk_dir / "chunked.json"
    chunker.save_chunks(all_chunks, output_file)
    print(f"[INFO] 총 {len(all_chunks)}개의 chunk 저장 완료: {output_file}")

if __name__ == "__main__":
    run()
