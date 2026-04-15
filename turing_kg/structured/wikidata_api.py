from __future__ import annotations

import json
import time
from typing import Any

import requests

from ..config import USER_AGENT


WIKIDATA_API = "https://www.wikidata.org/w/api.php"


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
    languages: str = "zh|en",
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
    for lang in ("zh", "en"):
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
    return {"root_qid": root_qid, "entities": entities, "item_edges": edges, "claims": claims}
