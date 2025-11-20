from bs4 import BeautifulSoup

def parse_html(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("title").get_text(strip=True) if soup.find("title") else "Untitled"
    text = soup.get_text(" ", strip=True)  # 연속공백 정리

    return {
        "url": url,
        "title": title,
        "text": text,
        "raw_html": html
    }

