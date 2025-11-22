def is_low_quality(text: str, min_length: int = 15) -> bool:
    if len(text.strip()) < min_length:
        return True
    if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.5:
        return True
    return False
