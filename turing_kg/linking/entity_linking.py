"""实体链接：Wikidata wbsearchentities + 候选重排序（标签与上下文）。"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

from ..config import USER_AGENT

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# 与「艾伦·图灵 / Alan Turing」易混的概念实体（Wikidata）
_QID_TURING_MACHINE = "Q163310"
_QID_TURING_AWARD = "Q185667"
_QID_TURING_PERSON = "Q7251"

# 在 mention 过短、或被 NER 从复合词中切开时，仅靠搜索容易误指到人物或其它条目；
# 本函数在 entity_map 与网络搜索**之前**执行，用句子上下文定界。
_TURING_EN_MACHINE = re.compile(
    r"\b(?:universal\s+)?turing\s+machines?\b|"
    r"\buniversal\s+turing\b|"
    r"\bnon-?deterministic\s+turing\b|"
    r"\bprobabilistic\s+turing\b",
    re.IGNORECASE,
)
_TURING_EN_AWARD = re.compile(
    r"\bturing\s+award\b|"
    r"\b(?:a\.?\s*)?m\.?\s*turing\s+award\b|"
    r"\bprize\s+\(?turing\)?\s+award\b",
    re.IGNORECASE,
)
# 人名强提示：同句出现全名或传记式书写时，裸露「Turing」倾向链到人物
_TURING_EN_PERSON = re.compile(
    r"\balan\s+turing\b|\balan\s+mathison\s+turing\b|\balan\s+m\.?\s*turing\b|"
    r"\bturing\s*[\(\,]|"
    r"^\s*turing\s*[\,]",
    re.IGNORECASE,
)


def _context_override_qid(mention: str, context: str) -> tuple[str, float] | None:
    """
    对「Turing/图灵」等高频歧义 mention，按上下文压过 entity_map 中的短别名（如「图灵」-> 人物）
    与误检索（如裸露 Turing -> 非 Q7251 条目）。

    说明：单句 `context` 下优先匹配更具体的「图灵机 / Turing machine / 图灵奖 / …」短语。
    """
    m = (mention or "").strip()
    if len(m) < 2:
        return None
    c = context or ""
    mlow = m.lower()

    if mlow == "turing":
        if _TURING_EN_MACHINE.search(c):
            return _QID_TURING_MACHINE, 1.0
        if _TURING_EN_AWARD.search(c):
            return _QID_TURING_AWARD, 1.0
        if _TURING_EN_PERSON.search(c):
            return _QID_TURING_PERSON, 1.0

    if m == "图灵" and "图灵机" in c:
        return _QID_TURING_MACHINE, 1.0

    return None


def _sleep_seconds_from_retry_after(resp: requests.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def _cache_path(kind: str, key: str) -> str:
    # key 里可能有中文；用轻量哈希避免路径问题
    import hashlib
    from pathlib import Path

    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    d = Path("data") / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{kind}_{h}.json")


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


def _char_ngrams(s: str, n: int) -> list[str]:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    if not s:
        return []
    # 对中文直接按字符切 ngram，对英文/数字也按字符；这样对错别字/拼写变化更稳
    if len(s) <= n:
        return [s]
    return [s[i : i + n] for i in range(0, len(s) - n + 1)]


def char_ngram_jaccard(a: str, b: str, *, n: int = 3) -> float:
    na = set(_char_ngrams(a, n))
    nb = set(_char_ngrams(b, n))
    if not na or not nb:
        return 0.0
    inter = len(na & nb)
    union = len(na | nb)
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
    # 轻量缓存：避免频繁重复搜索触发 429
    try:
        import json

        cpath = _cache_path("wbsearch", f"{language}|{limit}|{q}")
        from pathlib import Path

        p = Path(cpath)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass

    last: Exception | None = None
    for attempt in range(6):
        try:
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
            out = list(resp.json().get("search", []) or [])
            try:
                import json
                from pathlib import Path

                Path(cpath).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return out
        except requests.exceptions.HTTPError as e:
            last = e
            resp2 = getattr(e, "response", None)
            if resp2 is not None and resp2.status_code == 429 and attempt < 5:
                base = 1.5 * (attempt + 1)
                ra = _sleep_seconds_from_retry_after(resp2)
                sleep_s = max(base, ra or 0.0)
                sleep_s = min(max(sleep_s, 5.0), 60.0)
                time.sleep(sleep_s)
                continue
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt < 5:
                time.sleep(min(10.0, 1.2 * (attempt + 1)))
                continue
            raise
    if last is not None:
        raise last
    return []


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
        ngram_sim = char_ngram_jaccard(mention, label, n=3)
        position_prior = max(0.0, 0.12 - 0.015 * idx)
        score = label_bonus + 0.55 * overlap + 0.35 * ngram_sim + position_prior
        if score > best_score:
            best_score = score
            best_id = qid
    return best_id, best_score


def score_candidates(
    mention: str,
    context: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    为 Phase 1/协同链接输出“每个候选的局部打分”。
    """
    out: list[dict[str, Any]] = []
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
        ngram_sim = char_ngram_jaccard(mention, label, n=3)
        position_prior = max(0.0, 0.12 - 0.015 * idx)
        score = label_bonus + 0.55 * overlap + 0.35 * ngram_sim + position_prior
        out.append(
            {
                "qid": str(qid),
                "score": float(score),
                "breakdown": {
                    "label": label,
                    "description": desc,
                    "label_exact": bool(label and label.strip().lower() == mlow),
                    "context_jaccard": float(overlap),
                    "char_ngram_jaccard_3": float(ngram_sim),
                    "position_prior": float(position_prior),
                    "label_bonus": float(label_bonus),
                },
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


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
    ctx_hit = _context_override_qid(m, context or "")
    if ctx_hit:
        return ctx_hit
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
        # 基础节流：降低触发 Wikidata 搜索限流概率
        time.sleep(0.6)
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


def link_mention_with_candidates(
    mention: str,
    context: str,
    *,
    languages: tuple[str, ...] = ("zh", "en"),
    min_score: float = 0.12,
    entity_map_override: dict[str, dict] | None = None,
    candidate_limit_per_lang: int = 10,
    rerank_topn: int = 15,
) -> dict[str, Any]:
    """
    返回可审计的链接结果（含候选列表）。

    输出：
    - chosen_qid: str|None
    - chosen_score: float
    - candidates: list[dict]（Wikidata wbsearchentities 的原始候选去重后前 N 个）
    - override_hit: dict|None（若命中 entity_map）
    """
    m = mention.strip()
    if len(m) < 2:
        return {"chosen_qid": None, "chosen_score": 0.0, "candidates": [], "override_hit": None}

    ctx_hit = _context_override_qid(m, context or "")
    if ctx_hit:
        qid0, sc0 = ctx_hit
        return {
            "chosen_qid": qid0,
            "chosen_score": float(sc0),
            "candidates": [{"id": qid0, "label": "", "description": ""}],
            "override_hit": {
                "matched_key": m,
                "qid": qid0,
                "source": "context_disambiguation",
            },
            "candidate_scores": [
                {
                    "qid": qid0,
                    "score": float(sc0),
                    "breakdown": {
                        "label": "",
                        "description": "",
                        "source": "context_disambiguation",
                    },
                }
            ],
            "local_breakdown": {},
        }

    override_hit: dict[str, Any] | None = None
    if entity_map_override:
        keys = sorted(entity_map_override.keys(), key=len, reverse=True)
        low = m.lower()
        for k in keys:
            if len(k) < 2:
                continue
            if k.lower() in low or m == k:
                row = entity_map_override.get(k) or entity_map_override.get(k.lower())
                if row and row.get("wikidata_id"):
                    override_hit = {"matched_key": k, "qid": row["wikidata_id"]}
                    return {
                        "chosen_qid": row["wikidata_id"],
                        "chosen_score": 1.0,
                        "candidates": [
                            {
                                "id": row["wikidata_id"],
                                "label": row.get("label", "") if isinstance(row, dict) else "",
                                "description": row.get("description", "") if isinstance(row, dict) else "",
                            }
                        ],
                        "override_hit": override_hit,
                    }

    all_cands: list[dict[str, Any]] = []
    for lang in languages:
        time.sleep(0.6)
        cands = wb_search_entities(m, language=lang, limit=int(candidate_limit_per_lang))
        all_cands.extend(cands)

    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in all_cands:
        i = c.get("id")
        if i and i not in seen:
            seen.add(i)
            uniq.append(c)

    scored = score_candidates(m, context, uniq[: int(rerank_topn)])
    qid = scored[0]["qid"] if scored else None
    sc = float(scored[0]["score"]) if scored else 0.0
    chosen_qid: str | None = None
    if qid and (sc >= min_score or len(uniq) == 1):
        chosen_qid = qid

    return {
        "chosen_qid": chosen_qid,
        "chosen_score": float(sc),
        "candidates": uniq[: int(rerank_topn)],
        "override_hit": override_hit,
        "candidate_scores": scored,
        "local_breakdown": (scored[0]["breakdown"] if scored else {}),
    }
