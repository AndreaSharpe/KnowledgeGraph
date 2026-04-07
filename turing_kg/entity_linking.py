"""实体链接：Wikidata wbsearchentities + 候选重排序（标签与上下文）。"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

from .config import USER_AGENT

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def _token_set(s: str) -> set[str]:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s.lower(), flags=re.UNICODE)
    return {t for t in s.split() if len(t) > 1}


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def wb_search_entities(
    query: str,
    *,
    language: str = "en",
    limit: int = 8,
    timeout: tuple[float, float] = (20.0, 60.0),
) -> list[dict[str, Any]]:
    q = query.strip()
    if len(q) < 2:
        return []
    resp = requests.get(
        WIKIDATA_API,
        params={
            "action": "wbsearchentities",
            "search": q,
            "language": language,
            "format": "json",
            "limit": limit,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return list(resp.json().get("search", []) or [])


def rank_candidates(
    mention: str,
    context: str,
    candidates: list[dict[str, Any]],
) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = -1.0
    mlow = mention.strip().lower()
    for idx, c in enumerate(candidates):
        qid = c.get("id")
        if not qid or not str(qid).startswith("Q"):
            continue
        label = (c.get("label") or "").strip()
        desc = (c.get("description") or "").strip()
        label_bonus = 0.4 if label.lower() == mlow else 0.0
        if label and mlow in label.lower():
            label_bonus = max(label_bonus, 0.22)
        overlap = jaccard(context, f"{label} {desc}")
        position_prior = max(0.0, 0.12 - 0.015 * idx)
        score = label_bonus + 0.65 * overlap + position_prior
        if score > best_score:
            best_score = score
            best_id = qid
    return best_id, best_score


def link_mention_to_qid(
    mention: str,
    context: str,
    *,
    languages: tuple[str, ...] = ("zh", "en"),
    min_score: float = 0.12,
    entity_map_override: dict[str, dict] | None = None,
) -> tuple[str | None, float]:
    m = mention.strip()
    if len(m) < 2:
        return None, 0.0
    if entity_map_override:
        keys = sorted(entity_map_override.keys(), key=len, reverse=True)
        low = m.lower()
        for k in keys:
            if len(k) < 2:
                continue
            if k.lower() in low or m == k:
                row = entity_map_override.get(k) or entity_map_override.get(k.lower())
                if row and row.get("wikidata_id"):
                    return row["wikidata_id"], 1.0

    all_cands: list[dict[str, Any]] = []
    for lang in languages:
        time.sleep(0.25)
        cands = wb_search_entities(m, language=lang, limit=10)
        all_cands.extend(cands)

    if not all_cands:
        return None, 0.0

    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in all_cands:
        i = c.get("id")
        if i and i not in seen:
            seen.add(i)
            uniq.append(c)

    qid, sc = rank_candidates(m, context, uniq[:15])
    if not qid:
        return None, sc
    if sc >= min_score:
        return qid, sc
    if len(uniq) == 1:
        return qid, sc
    return None, sc
