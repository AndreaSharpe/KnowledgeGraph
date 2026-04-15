"""关系抽取相关 JSON 配置加载（schema / allowlist）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_relation_allowlist(project_root: Path) -> dict[str, Any]:
    p = project_root / "sources" / "relation_allowlist.json"
    if not p.is_file():
        return {"by_seed_type": {}, "by_seed_id_override": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def load_relation_schema(project_root: Path) -> dict[str, Any]:
    p = project_root / "sources" / "relation_schema.json"
    if not p.is_file():
        return {"relations": []}
    return json.loads(p.read_text(encoding="utf-8"))


def prop_label_for(project_root: Path, prop_id: str) -> str:
    data = load_relation_schema(project_root)
    for r in data.get("relations") or []:
        if str(r.get("prop_id")) == prop_id:
            return str(r.get("label") or prop_id)
    return prop_id


def load_relation_thresholds(project_root: Path) -> dict[str, Any]:
    p = project_root / "sources" / "relation_thresholds.json"
    if not p.is_file():
        return {"default_threshold": 0.75, "by_prop_id": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def threshold_for_prop(thresholds: dict[str, Any], prop_id: str) -> float:
    by = thresholds.get("by_prop_id") or {}
    if prop_id in by:
        return float(by[prop_id])
    return float(thresholds.get("default_threshold", 0.75))


def labels_space_for_seed(allowlist: dict[str, Any], *, seed_type: str, seed_id: str) -> list[str]:
    """返回该 seed 的关系预测空间（用于 mask / DS 标签空间）。"""
    override = allowlist.get("by_seed_id_override") or {}
    if seed_id in override and override[seed_id]:
        return [str(x) for x in override[seed_id]]
    by_type = allowlist.get("by_seed_type") or {}
    return [str(x) for x in by_type.get(seed_type, [])]
