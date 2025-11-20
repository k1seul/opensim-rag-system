import time
import random
import requests
from .config import HEADERS

def fetch_html(url: str) -> str:
    time.sleep(random.uniform(0.8, 1.8))
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"[ERROR] {r.status_code} - {url}")
        return ""

    return r.text

