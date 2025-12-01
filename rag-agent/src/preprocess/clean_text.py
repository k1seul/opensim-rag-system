import re
from bs4 import BeautifulSoup
import unicodedata
import html

def clean_text(raw_html: str) -> str:
    """
    HTML, URL, escape sequences 등을 정리하여 깨끗한 텍스트 반환
    """
    # 1. Remove HTML tags, keep spaces
    text = BeautifulSoup(raw_html, "html.parser").get_text(" ")

    # 2. Decode HTML entities (&amp; -> &)
    text = html.unescape(text)

    # 3. Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # 4. Remove common escape sequences like \\\"
    text = re.sub(r'\\+', '', text)

    # 5. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove isolated 'n' characters repeated from HTML parsing noise
    text = re.sub(r'\b[nN]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text