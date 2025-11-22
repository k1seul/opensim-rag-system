import re
from bs4 import BeautifulSoup
import unicodedata


def clean_text(html: str) -> str:
    # Remove HTML tags
    text = BeautifulSoup(html, "html.parser").get_text(" ")

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove escape sequences and excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
