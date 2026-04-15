from __future__ import annotations

from dataclasses import dataclass

import requests

from ..config import USER_AGENT


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


def fetch_seed_excerpts(seeds: list[dict]) -> list[WikiChunk]:
    """
    通用多 seed 维基摘要抓取（复用 _fetch_extract）。

    约定：
    - 每个 seed 可提供 anchors_zh / anchors_en；本函数取第一个 anchor 作为维基条目标题尝试抓取。
    - 若抓取失败则跳过（避免因为某个 seed 标题不匹配导致全局失败）。
    """
    out: list[WikiChunk] = []
    for s in seeds or []:
        az = list(s.get("anchors_zh", []) or [])
        ae = list(s.get("anchors_en", []) or [])
        if az:
            title = str(az[0]).strip()
            if title:
                try:
                    out.append(_fetch_extract("zh", title))
                except Exception:
                    pass
        if ae:
            title = str(ae[0]).strip()
            if title:
                try:
                    out.append(_fetch_extract("en", title))
                except Exception:
                    pass
    return out
