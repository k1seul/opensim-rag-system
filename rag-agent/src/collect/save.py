import os
import json

RAW_PATH = "data/raw"

os.makedirs(RAW_PATH, exist_ok=True)

def save_raw(data: dict):
    filename = data["title"].replace(" ", "_").replace("/", "_")[:100] + ".json"
    file_path = os.path.join(RAW_PATH, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {file_path}")

