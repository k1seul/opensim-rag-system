from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .fetch import fetch_html

def crawl_all_links(start_url: str):
    visited = set()
    queue = [start_url]
    collected = []

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        
        visited.add(url)

        html = fetch_html(url)
        if not html:
            continue

        collected.append((url, html))
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if "/wiki/spaces/OpenSim/" in link and link not in visited:
                queue.append(link)

    return collected

