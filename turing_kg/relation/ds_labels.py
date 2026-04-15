"""
为每个 bag 生成远程监督标签 data/curated/ds_labels.jsonl。

规则：若 Wikidata 上存在 (subject_qid, prop_id, object_qid) 且 prop_id 属于该 seed 的 allowlist，
则 prop_id 记入 labels_pos。labels_space 来自 sources/relation_allowlist.json。

依赖：结构化查询复用 turing_kg.structured.wikidata_api.wbgetentities + iter_claim_item_edges。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..structured.wikidata_api import iter_claim_item_edges, wbgetentities
from .config_loaders import labels_space_for_seed, load_relation_allowlist


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL（每行一个 JSON 对象）。"""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _fetch_subject_edges(subject_qids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """subject_qid -> [(prop_id, object_qid), ...]（仅 wikibase-item 值）。"""
    qids = sorted(subject_qids)
    out_map: dict[str, list[tuple[str, str]]] = {q: [] for q in qids}
    if not qids:
        return out_map

    # wbgetentities 已内建分块与缓存
    for i in range(0, len(qids), 20):
        chunk = qids[i : i + 20]
        ents = wbgetentities(chunk, props="claims")
        for qid in chunk:
            ent = ents.get(qid) or {}
            claims = ent.get("claims") or {}
            out_map[qid] = iter_claim_item_edges(claims)
    return out_map


def build_ds_label_rows(
    project_root: Path,
    bags: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    若 bags 为 None，则从 data/curated/bags.jsonl 读取。
    """
    allowlist = load_relation_allowlist(project_root)
    if bags is None:
        bags_path = project_root / "data" / "curated" / "bags.jsonl"
        bags = read_jsonl(bags_path)

    subject_qids = {str(b.get("subject_qid") or "").strip() for b in bags}
    subject_qids = {q for q in subject_qids if q.startswith("Q")}
    edges_by_subject = _fetch_subject_edges(subject_qids)

    rows: list[dict[str, Any]] = []
    for bag in bags:
        bag_id = str(bag.get("bag_id") or "")
        seed_id = str(bag.get("seed_id") or "")
        seed_type = str(bag.get("seed_type") or "Unknown")
        subject_qid = str(bag.get("subject_qid") or "").strip()
        object_qid = str(bag.get("object_qid") or "").strip()
        if not bag_id or not subject_qid.startswith("Q") or not object_qid.startswith("Q"):
            continue

        space = labels_space_for_seed(allowlist, seed_type=seed_type, seed_id=seed_id)
        space_set = set(space)
        edges = edges_by_subject.get(subject_qid, [])

        pos_by_prop: dict[str, str] = {}
        for prop_id, tgt_qid in edges:
            if tgt_qid != object_qid:
                continue
            if prop_id not in space_set:
                continue
            pos_by_prop[prop_id] = f"wikidata ({subject_qid},{prop_id},{object_qid})"

        labels_pos = sorted(pos_by_prop.keys())
        matched = [{"prop_id": p, "evidence": pos_by_prop[p]} for p in labels_pos]

        rows.append(
            {
                "bag_id": bag_id,
                "seed_id": seed_id,
                "seed_type": seed_type,
                "subject_qid": subject_qid,
                "object_qid": object_qid,
                "labels_pos": labels_pos,
                "labels_space": space,
                "label_source": {
                    "kb": "wikidata",
                    "matched_triples": matched,
                },
                "negatives": {
                    "type_incompatible_props": [],
                    "hard_negative_note": "",
                },
            }
        )

    rows.sort(key=lambda r: r.get("bag_id", ""))
    return rows


def write_ds_labels_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> Path:
    out = project_root / "data" / "curated" / "ds_labels.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def build_and_write_ds_labels(project_root: Path) -> tuple[Path, int]:
    rows = build_ds_label_rows(project_root)
    path = write_ds_labels_jsonl(project_root, rows)
    return path, len(rows)
