import re
import requests
from bs4 import BeautifulSoup
from typing import Iterable, List, Optional

def fetch_wiki_page(topic: str, verbose: bool = False) -> str:
    url = f"https://en.wikipedia.org/wiki/{topic}"
    headers = {"User-Agent": "Mozilla/5.0"}
    if verbose:
        print(f"[FETCH] {url}")
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        content = soup.find("div", id="mw-content-text")
        if not content:
            return topic
        for elem in content.find_all(["sup", "table", "style", "script", "figure"]):
            elem.decompose()
        text = content.get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()[:50000]
    except Exception as e:
        if verbose:
            print(f"[FETCH] failed {e}")
        return topic

def load_topics(topics_file: str, limit: Optional[int] = None) -> List[str]:
    with open(topics_file, "r", encoding="utf-8") as f:
        topics = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return topics[:limit] if limit else topics
