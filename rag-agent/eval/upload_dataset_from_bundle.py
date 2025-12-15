# upload_dataset_from_bundle.py
import os, json
from pathlib import Path
from langsmith import Client

version = "pro"

def main():
    bundle_path = os.getenv("EVAL_BUNDLE_PATH", f"eval/dataset/rag_{version}.json")
    dataset_name = os.getenv("EVAL_DATASET_NAME", f"opensim-rag-eval-{version}")

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))

    client = Client()

    try:
        client.read_dataset(dataset_name=dataset_name)
        print(f"[INFO] Dataset exists: {dataset_name}")
    except Exception:
        client.create_dataset(
            dataset_name=dataset_name,
            description="OpenSim RAG eval dataset from a single JSON bundle (answers embedded).",
        )
        print(f"[INFO] Created dataset: {dataset_name}")

    examples = []
    for item in bundle["items"]:
        contexts = [c.get("text", "") for c in item["retrieval"]["contexts"]]

        examples.append({
            "inputs": {
                "question": item["question"],
                "answer": item.get("answer", ""),  
                "contexts": contexts,
                "meta": {
                    "id": item.get("id"),
                    "top_k": item["retrieval"].get("top_k"),
                    "use_web": item["retrieval"].get("use_web"),
                    # ✅ 있으면 같이 업로드 (없어도 에러 안남)
                    "answer_md_path": item.get("answer_md_path"),
                },
            },
            "outputs": {}
        })

    client.create_examples(dataset_name=dataset_name, examples=examples)
    print(f"[DONE] Uploaded {len(examples)} examples to {dataset_name}")

if __name__ == "__main__":
    main()
