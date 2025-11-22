# src/preprocess/extract_main_paragraphs.py
import re

def extract_main_paragraphs(text, min_length=100):
    """Extract paragraphs that are likely to be meaningful content."""
    # Split into paragraphs by line breaks or double newlines
    paragraphs = re.split(r'\n\n+|\n', text)

    # Keep only paragraphs longer than min_length
    main_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

    # Join back into single text
    return ' '.join(main_paragraphs)

# Usage example
if __name__ == '__main__':
    sample_text = """Short line\n\nThis is a real paragraph explaining OpenSim usage. It contains detailed steps.\nAnother short line."""
    result = extract_main_paragraphs(sample_text)
    print(result)
