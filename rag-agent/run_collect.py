from src.collect.crawler import crawl_all_links
from src.collect.parser import parse_html
from src.collect.save import save_raw
from src.collect.config import BASE_URL
from tqdm import tqdm

pages = crawl_all_links(BASE_URL)

for url, html in tqdm(pages, desc="Collecting & Saving Pages", unit="page"):
    data = parse_html(url, html)
    save_raw(data)

