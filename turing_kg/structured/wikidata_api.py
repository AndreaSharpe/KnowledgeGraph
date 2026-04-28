from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import requests

from ..config import USER_AGENT


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS_URL = "https://query.wikidata.org/sparql"

_URI_TAIL_Q_P = re.compile(r"(Q\d+|P\d+)\s*$")


def _sleep_seconds_from_retry_after(resp: requests.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def _cache_dir() -> str:
    return "data/cache"


def _cache_key(ids_chunk: str, props: str, languages: str) -> str:
    # 文件名需稳定且尽量短；ids_chunk 已是 'Q..|P..' 形式
    safe_ids = ids_chunk.replace("|", "_")
    safe_props = props.replace("|", "_")
    safe_lang = languages.replace("|", "_")
    return f"wbgetentities_{safe_ids}__{safe_props}__{safe_lang}.json"


def wbgetentities(
    ids: list[str],
    *,
    props: str = "labels|claims",
    languages: str = "zh-hans|zh|en",
    timeout: tuple[float, float] = (45.0, 180.0),
    retries: int = 5,
) -> dict[str, dict]:
    if not ids:
        return {}
    merged: dict[str, dict] = {}
    # 保守分块：降低一次请求的实体数量，有助于避免 429。
    for i in range(0, len(ids), 20):
        chunk = "|".join(ids[i : i + 20])

        # 轻量本地缓存：避免反复运行时触发 Wikidata 限流（429）。
        # 缓存以 chunk+props+languages 为粒度；结构化层的数据变化频率低，足以满足课程项目可复现。
        try:
            from pathlib import Path

            cdir = Path(_cache_dir())
            cdir.mkdir(parents=True, exist_ok=True)
            cpath = cdir / _cache_key(chunk, props, languages)
            if cpath.is_file():
                data = json.loads(cpath.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged.update(data)
                    time.sleep(0.05)
                    continue
        except Exception:
            # 缓存失败不影响主流程
            pass

        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(
                    WIKIDATA_API,
                    params={
                        "action": "wbgetentities",
                        "ids": chunk,
                        "format": "json",
                        "props": props,
                        "languages": languages,
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=timeout,
                )
                resp.raise_for_status()
                ents = resp.json().get("entities", {}) or {}
                merged.update(ents)
                try:
                    # 写入缓存（只缓存 entities）
                    from pathlib import Path

                    cdir = Path(_cache_dir())
                    cdir.mkdir(parents=True, exist_ok=True)
                    cpath = cdir / _cache_key(chunk, props, languages)
                    cpath.write_text(json.dumps(ents, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                last = None
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                last = e
                if attempt + 1 < retries:
                    # 遇到 429 时尊重 Retry-After，并做指数退避（带上限）
                    sleep_s = 2.0 * (attempt + 1)
                    if isinstance(e, requests.exceptions.HTTPError) and getattr(e, "response", None) is not None:
                        resp2 = e.response
                        if resp2 is not None and resp2.status_code == 429:
                            ra = _sleep_seconds_from_retry_after(resp2)
                            if ra is not None:
                                sleep_s = max(sleep_s, ra)
                            sleep_s = min(max(sleep_s, 10.0), 90.0)
                    time.sleep(sleep_s)
        if last is not None:
            # 若多次重试仍遇到 429，则返回已获取的部分结果（避免整条流水线失败）。
            if isinstance(last, requests.exceptions.HTTPError) and getattr(last, "response", None) is not None:
                if last.response is not None and last.response.status_code == 429:
                    return merged
            raise last
        # 基础节流：降低被限流概率
        time.sleep(1.8)
    return merged


def pick_label(ent: dict[str, Any]) -> str:
    labels = ent.get("labels") or {}
    # 经典中文展示：优先简体（zh-hans/zh-cn），再退回 zh（可能是繁体），最后 en
    for lang in ("zh-hans", "zh-cn", "zh", "en"):
        cell = labels.get(lang)
        if isinstance(cell, dict) and cell.get("value"):
            return str(cell["value"])
    for cell in labels.values():
        if isinstance(cell, dict) and cell.get("value"):
            return str(cell["value"])
    return str(ent.get("id", ""))


def format_datavalue(dv: dict[str, Any]) -> str | None:
    t = dv.get("type")
    val = dv.get("value")
    if t == "time" and isinstance(val, dict):
        return str(val.get("time", val))
    if t == "string":
        return str(val)
    if t == "monolingualtext" and isinstance(val, dict):
        return str(val.get("text", val))
    if t == "quantity" and isinstance(val, dict):
        return str(val.get("amount", val))
    if t == "globecoordinate" and isinstance(val, dict):
        lat, lon = val.get("latitude"), val.get("longitude")
        if lat is not None and lon is not None:
            return f"{lat},{lon}"
    return None


def iter_claim_item_edges(claims: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for prop_id, statements in claims.items():
        if not str(prop_id).startswith("P"):
            continue
        if not isinstance(statements, list):
            continue
        for st in statements:
            snak = st.get("mainsnak") or {}
            if snak.get("snaktype") != "value":
                continue
            dv = snak.get("datavalue") or {}
            if dv.get("type") != "wikibase-entityid":
                continue
            tid = (dv.get("value") or {}).get("id")
            if isinstance(tid, str) and tid.startswith("Q"):
                out.append((prop_id, tid))
    return out


def iter_claim_literal_snippets(
    claims: dict[str, Any],
    *,
    only_props: set[str] | None = None,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for prop_id, statements in claims.items():
        if not str(prop_id).startswith("P"):
            continue
        if only_props is not None and prop_id not in only_props:
            continue
        if not isinstance(statements, list):
            continue
        for st in statements:
            snak = st.get("mainsnak") or {}
            if snak.get("snaktype") != "value":
                continue
            dv = snak.get("datavalue") or {}
            if dv.get("type") == "wikibase-entityid":
                continue
            text = format_datavalue(dv)
            if text:
                out.append((prop_id, text))
    return out


def load_root_neighborhood(root_qid: str) -> dict[str, Any]:
    base = wbgetentities([root_qid], props="labels|claims")
    root_ent = base.get(root_qid) or {}
    claims = root_ent.get("claims") or {}
    edges = iter_claim_item_edges(claims)
    p_ids = {p for p, _ in edges}
    q_ids = {q for _, q in edges}
    want = sorted(p_ids | q_ids)
    # label enrich 是“锦上添花”；遇到限流时不应导致整个流水线失败
    try:
        extra = wbgetentities(want, props="labels") if want else {}
    except requests.exceptions.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 429:
            extra = {}
        else:
            raise
    entities = {**base, **extra}

    # 类型推断需要 P31 等 claim。为了让图里节点能显示 Person/Location/...，
    # 这里“尽力”为邻域里的 Q 节点补取 claims（若限流/断网则跳过，不影响主流程）。
    # 注意：prop 节点（Pxxx）不需要 claims，只需 label。
    try:
        if q_ids:
            extra_claims = wbgetentities(sorted(q_ids), props="labels|claims")
            entities.update(extra_claims or {})
    except Exception:
        pass
    return {"root_qid": root_qid, "entities": entities, "item_edges": edges, "claims": claims}


def _entity_uri_to_q_or_p(uri: str) -> str:
    m = _URI_TAIL_Q_P.search((uri or "").rstrip("/"))
    return m.group(1) if m else ""


def sparql_select_json(query: str, *, timeout: tuple[float, float] = (30.0, 120.0)) -> list[dict[str, Any]]:
    """
    在 Wikidata Query Service 上执行 SPARQL，返回 bindings 列表（失败时返回空列表）。
    使用轻量文件缓存，避免构建时反复打满 WDQS 配额。
    """
    q = (query or "").strip()
    if not q:
        return []
    h = hashlib.sha1(q.encode("utf-8")).hexdigest()[:20]
    try:
        from pathlib import Path

        cpath = Path(_cache_dir()) / f"wdqs_{h}.json"
        if cpath.is_file():
            data = json.loads(cpath.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    for attempt in range(3):
        try:
            time.sleep(0.35 * (attempt + 1))
            resp = requests.get(
                WDQS_URL,
                params={"query": q, "format": "json"},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/sparql-results+json",
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                time.sleep(min(15.0, 3.0 * (attempt + 1)))
                continue
            resp.raise_for_status()
            j = resp.json() or {}
            out = j.get("results", {}).get("bindings") or []
            if not isinstance(out, list):
                out = []
            try:
                from pathlib import Path

                cpath = Path(_cache_dir()) / f"wdqs_{h}.json"
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            return out
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return []


def fetch_incoming_item_edges(
    object_qid: str,
    allowed_props: tuple[str, ...],
    *,
    limit: int = 40,
) -> list[tuple[str, str, str]]:
    """
    查询「其它 Wikidata 条目作为主体、值为 object_qid 的**直接陈述**（wdt: 真值）」
    返回 (subject_qid, prop_id, object_qid)，按 LIMIT 截断。用于为概念/奖项补充入向关联。
    """
    oq = (object_qid or "").strip()
    if not oq.startswith("Q") or not allowed_props:
        return []
    n = max(1, min(int(limit), 500))
    pvals = " ".join(f"wdt:{p}" for p in allowed_props)
    # DISTINCT 避免同一条在部分属性下被重复（极少见）
    q = f"""
SELECT DISTINCT ?s ?p WHERE {{
  VALUES ?p {{ {pvals} }}
  ?s ?p wd:{oq} .
  FILTER( STRSTARTS( STR( ?s ), "http://www.wikidata.org/entity/Q" ) )
}}
ORDER BY ?s ?p
LIMIT {n}
"""
    rows: list[dict[str, Any]] = sparql_select_json(q)
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for b in rows:
        s_uri = (b.get("s") or {}).get("value") or ""
        p_uri = (b.get("p") or {}).get("value") or ""
        s_id = _entity_uri_to_q_or_p(s_uri)
        p_id = _entity_uri_to_q_or_p(p_uri)
        if not s_id.startswith("Q") or not p_id.startswith("P"):
            continue
        key = (s_id, p_id, oq)
        if key in seen:
            continue
        seen.add(key)
        out.append((s_id, p_id, oq))
    return out
