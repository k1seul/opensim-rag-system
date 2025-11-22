from datetime import datetime
import re


def extract_metadata(raw_text: str) -> dict:
    metadata = {}

    urls = re.findall(r"https?://[^\s]+", raw_text)
    if urls:
        metadata["urls"] = urls

    metadata["timestamp"] = datetime.now().isoformat()
    return metadata
