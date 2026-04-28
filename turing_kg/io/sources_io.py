"""加载项目配置：实体别名表、参考文献、NER/链接参数。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_entity_map(path: Path) -> dict[str, dict]:
    by_alias: dict[str, dict] = {}
    if not path.is_file():
        return by_alias
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            alias = (row.get("alias") or "").strip()
            qid = (row.get("wikidata_id") or "").strip()
            kind = (row.get("kind") or "").strip()
            if alias and qid:
                rowd = {"wikidata_id": qid, "kind": kind}
                by_alias[alias] = rowd
                by_alias[alias.lower()] = rowd
    return by_alias


def load_entity_kind_by_qid(path: Path) -> dict[str, str]:
    """
    从 sources/entity_map.csv 提取 QID -> kind（粗类型）映射。

    设计目的：
    - Neo4j 节点 :LABEL 的最小可行类型来源（离线优先，避免联网）。
    - 同一 QID 可能对应多个 alias；kind 取第一个非空值即可。
    """
    by_qid: dict[str, str] = {}
    if not path.is_file():
        return by_qid
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = (row.get("wikidata_id") or "").strip()
            kind = (row.get("kind") or "").strip()
            if not qid.startswith("Q") or not kind:
                continue
            if qid not in by_qid:
                by_qid[qid] = kind
    return by_qid


def load_bibliography(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for item in data:
        key = item.get("key")
        if isinstance(key, str):
            out[key] = item
    return out
