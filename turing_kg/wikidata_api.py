from __future__ import annotations

import time
from typing import Any

import requests

from .config import USER_AGENT


WIKIDATA_API = "https://www.wikidata.org/w/api.php"


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
    for i in range(0, len(ids), 50):
        chunk = "|".join(ids[i : i + 50])
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
                merged.update(resp.json().get("entities", {}))
                last = None
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                last = e
                if attempt + 1 < retries:
                    time.sleep(2.0 * (attempt + 1))
        if last is not None:
            raise last
        time.sleep(0.3)
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
    extra = wbgetentities(want, props="labels") if want else {}
    entities = {**base, **extra}
    return {"root_qid": root_qid, "entities": entities, "item_edges": edges, "claims": claims}
