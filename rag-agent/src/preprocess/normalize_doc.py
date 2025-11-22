import json
from typing import Dict, Any


def normalize_doc(
    doc_id: str, source: str, content: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source": source,
        "content": content,
        "metadata": metadata,
    }
