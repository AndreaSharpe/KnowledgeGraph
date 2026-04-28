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
            # 经典展示：优先中文关系名，其次 schema label（英文）
            zh = r.get("label_zh")
            if isinstance(zh, str) and zh.strip():
                return zh.strip()
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


def min_non_na_prob_for_seed_type(thresholds: dict[str, Any], seed_type: str) -> float:
    """
    经典 DS 多分类下：只有当 top-1 非 NA 且 prob >= tau 才导出为关系边，否则视为 NA。

    支持两种配置：
    - v2: {default_min_non_na_prob, by_seed_type{Person:..}}
    - v1 兼容：沿用 default_threshold 作为 tau
    """
    by_type = thresholds.get("by_seed_type") or {}
    if seed_type in by_type:
        return float(by_type[seed_type])
    if "default_min_non_na_prob" in thresholds:
        return float(thresholds.get("default_min_non_na_prob", 0.5))
    # fallback for legacy file
    return float(thresholds.get("default_threshold", 0.75))


def labels_space_for_seed(allowlist: dict[str, Any], *, seed_type: str, seed_id: str) -> list[str]:
    """返回该 seed 的关系预测空间（用于 mask / DS 标签空间）。"""
    override = allowlist.get("by_seed_id_override") or {}
    if seed_id in override and override[seed_id]:
        return [str(x) for x in override[seed_id]]
    by_type = allowlist.get("by_seed_type") or {}
    return [str(x) for x in by_type.get(seed_type, [])]
