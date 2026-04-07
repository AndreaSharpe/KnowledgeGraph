"""加载项目配置：实体别名表、参考文献、NER/链接参数。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

#加载别名表
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

#加载参考文献
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
