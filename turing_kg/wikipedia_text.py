from __future__ import annotations

from dataclasses import dataclass

import requests

from .config import USER_AGENT


@dataclass(frozen=True)
class WikiChunk:
    lang: str
    title: str
    url: str
    text: str


def _fetch_extract(lang: str, title: str) -> WikiChunk:
    host = f"https://{lang}.wikipedia.org/w/api.php"
    r = requests.get(
        host,
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "explaintext": "true",
            "exintro": "true",
            "inprop": "url",
            "redirects": "1",
            "titles": title,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=(30, 120),
    )
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("维基 API 未返回页面")
    page = next(iter(pages.values()))
    if page.get("missing"):
        raise ValueError(f"无此页面：{lang}:{title}")
    text = page.get("extract") or ""
    url = page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return WikiChunk(lang=lang, title=title, url=url, text=text.strip())


def fetch_turing_excerpts() -> list[WikiChunk]:
    """中英文「艾伦·图灵」条目摘要（维基 API）。"""
    return [
        _fetch_extract("zh", "艾伦·图灵"),
        _fetch_extract("en", "Alan Turing"),
    ]
