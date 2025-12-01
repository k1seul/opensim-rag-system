from bs4 import BeautifulSoup
import html as html_module
import unicodedata
import re

def parse_html(url: str, raw_html: str) -> dict:
    """
    HTML에서 제목과 본문만 추출하고, 텍스트를 정리하여 반환
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. 제목 추출
    title = soup.find("title").get_text(strip=True) if soup.find("title") else "Untitled"

    # 2. 본문 container 선택 (OpenSim 문서 기준)
    main_div = soup.select_one(".wiki-content")  # 실제 클래스 확인 필요
    if main_div is None:
        main_div = soup  # fallback: 전체 HTML

    # 3. 텍스트 정리
    text = main_div.get_text(" ")
    text = html_module.unescape(text)  # 내장 html 모듈 명시적으로 사용
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\\+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return {
        "url": url,
        "title": title,
        "text": text,
        "raw_html": raw_html
    }
