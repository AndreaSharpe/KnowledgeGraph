"""
从中间层构造 MIL 所需的 bag（data/curated/bags.jsonl）。

输入：
- data/processed/sentences.jsonl
- data/processed/routing.jsonl
- data/curated/resolved.jsonl（仅 chosen_qid 为 Q… 且非自环）

门控：句子 zh_ratio >= min_zh_ratio（默认 0.3），与 IMPLEMENTATION 文档一致。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..attribution.seed_config import SeedSpec, load_seed_entities
from .lang import zh_ratio


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _seed_type_map(seeds: list[SeedSpec]) -> dict[str, str]:
    return {s.seed_id: (s.type or "").strip() or "Unknown" for s in seeds}


def _routing_for_seed(routing_row: dict[str, Any] | None, seed_id: str) -> dict[str, Any] | None:
    if not routing_row:
        return None
    assigned = routing_row.get("assigned") or []
    for a in assigned:
        if str(a.get("seed_id", "")) == seed_id:
            return {
                "score": float(a.get("score") or 0.0),
                "reasons": a.get("reasons") if isinstance(a.get("reasons"), dict) else {},
            }
    return None


def build_bags(
    project_root: Path,
    *,
    min_zh_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    """
    读取已落盘的 sentences / routing / resolved，输出 bags 行列表（每行对应文档 bags.jsonl 的一条记录）。
    """
    processed = project_root / "data" / "processed"
    curated = project_root / "data" / "curated"

    sentences_path = processed / "sentences.jsonl"
    routing_path = processed / "routing.jsonl"
    resolved_path = curated / "resolved.jsonl"

    seeds = load_seed_entities(project_root)
    st_map = _seed_type_map(seeds)

    sent_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _iter_jsonl(sentences_path):
        sid = str(row.get("source_id", ""))
        si = int(row.get("sentence_idx", -1))
        if sid and si >= 0:
            sent_map[(sid, si)] = row

    route_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _iter_jsonl(routing_path):
        sid = str(row.get("source_id", ""))
        si = int(row.get("sentence_idx", -1))
        if sid and si >= 0:
            route_map[(sid, si)] = row

    # bag_key -> 聚合结构
    bag_meta: dict[str, dict[str, Any]] = {}
    # (bag_key, sentence_idx) -> instance
    inst_cells: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"mentions": [], "ner_labels": []}
    )

    for row in _iter_jsonl(resolved_path):
        chosen = str(row.get("chosen_qid") or "").strip()
        if not chosen.startswith("Q"):
            continue
        seed_qid = str(row.get("seed_qid") or "").strip()
        if chosen == seed_qid:
            continue

        source_id = str(row.get("source_id", ""))
        seed_id = str(row.get("seed_id", ""))
        if not source_id or not seed_id:
            continue

        sentence_idx = int(row.get("sentence_idx", -1))
        if sentence_idx < 0:
            continue

        sent_row = sent_map.get((source_id, sentence_idx))
        if not sent_row:
            continue
        sentence = str(sent_row.get("sentence") or "").strip()
        if zh_ratio(sentence) < min_zh_ratio:
            continue

        mention = str(row.get("mention") or "").strip()
        ner_label = str(row.get("ner_label") or "")

        bag_id = f"{source_id}|{seed_id}|{chosen}"
        if bag_id not in bag_meta:
            bag_meta[bag_id] = {
                "bag_id": bag_id,
                "source_id": source_id,
                "source_url": str(row.get("source_url") or ""),
                "source_label": str(row.get("source_label") or ""),
                "citation_key": str(row.get("citation_key") or ""),
                "seed_id": seed_id,
                "subject_qid": seed_qid,
                "seed_type": st_map.get(seed_id, "Unknown"),
                "object_qid": chosen,
            }

        cell_key = (bag_id, sentence_idx)
        ic = inst_cells[cell_key]
        if mention:
            ic["mentions"].append(mention)
        if ner_label:
            ic["ner_labels"].append(ner_label)

    # 组装 instances
    bags_out: list[dict[str, Any]] = []
    by_bag: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for (bag_id, sentence_idx), cell in inst_cells.items():
        meta = bag_meta.get(bag_id)
        if not meta:
            continue
        source_id = meta["source_id"]
        seed_id = meta["seed_id"]
        sent_row = sent_map.get((source_id, sentence_idx))
        if not sent_row:
            continue
        sentence = str(sent_row.get("sentence") or "").strip()
        rrow = route_map.get((source_id, sentence_idx))
        routing = _routing_for_seed(rrow, seed_id)

        mentions_unique = sorted(set(cell["mentions"]))
        ner_unique = sorted(set(cell["ner_labels"]))
        primary_mention = mentions_unique[0] if mentions_unique else ""
        primary_ner = ner_unique[0] if ner_unique else ""

        inst: dict[str, Any] = {
            "sentence_idx": sentence_idx,
            "sentence": sentence,
            "routing": routing,
            "mentions": {"object_mention": primary_mention, "ner_label": primary_ner or None},
        }
        by_bag[bag_id].append((sentence_idx, inst))

    for bag_id, meta in sorted(bag_meta.items(), key=lambda x: x[0]):
        pairs = sorted(by_bag.get(bag_id, []), key=lambda t: t[0])
        instances = [p[1] for p in pairs]
        if not instances:
            continue
        mention_set: set[str] = set()
        for (bid, _si), cell in inst_cells.items():
            if bid != bag_id:
                continue
            for m in cell.get("mentions") or []:
                if str(m).strip():
                    mention_set.add(str(m).strip())
        row_out = dict(meta)
        row_out["object_mention_set"] = sorted(mention_set)
        row_out["instances"] = instances
        bags_out.append(row_out)

    return bags_out


def write_bags_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> Path:
    out = project_root / "data" / "curated" / "bags.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def build_and_write_bags(project_root: Path, *, min_zh_ratio: float = 0.3) -> tuple[Path, int]:
    rows = build_bags(project_root, min_zh_ratio=min_zh_ratio)
    path = write_bags_jsonl(project_root, rows)
    return path, len(rows)
