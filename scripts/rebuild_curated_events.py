#!/usr/bin/env python3
"""
独立重建事件层（curated/events.jsonl），不跑全文构建、也不要求重跑 NER/EL。

输入：
- data/processed/sentences.jsonl
- data/processed/routing.jsonl
- data/curated/resolved.jsonl（可选，但强烈建议有；用于提供已链接实体作为论元候选）

输出（覆盖写入）：
- data/curated/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 curated/events.jsonl（事件抽取中间层）")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    root = args.project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from turing_kg.attribution.seed_config import load_seed_entities
    from turing_kg.extraction.event_patterns import extract_events_from_sentences
    from turing_kg.io.sources_io import load_entity_map

    processed = root / "data" / "processed"
    curated = root / "data" / "curated"
    curated.mkdir(parents=True, exist_ok=True)

    sentences_rows = _iter_jsonl(processed / "sentences.jsonl")
    routing_rows = _iter_jsonl(processed / "routing.jsonl")
    resolved_rows = _iter_jsonl(curated / "resolved.jsonl")

    if not sentences_rows or not routing_rows:
        raise RuntimeError("需要非空的 data/processed/sentences.jsonl 与 data/processed/routing.jsonl。")

    # sentence index map
    by_src_idx: dict[str, dict[int, str]] = defaultdict(dict)
    meta_by_source: dict[str, dict[str, str]] = {}
    for row in sentences_rows:
        sid = str(row.get("source_id") or "")
        si = int(row.get("sentence_idx", -1))
        if not sid or si < 0:
            continue
        by_src_idx[sid][si] = str(row.get("sentence") or "")
        if sid not in meta_by_source:
            meta_by_source[sid] = {
                "source_url": str(row.get("source_url") or ""),
                "source_label": str(row.get("source_label") or ""),
                "citation_key": str(row.get("citation_key") or ""),
            }

    # routing: (source_id, seed_id) -> set(sentence_idx)
    idx_by_src_seed: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in routing_rows:
        sid = str(row.get("source_id") or "")
        si = int(row.get("sentence_idx", -1))
        if not sid or si < 0:
            continue
        for a in row.get("assigned") or []:
            seed_id = str(a.get("seed_id") or "").strip()
            if seed_id:
                idx_by_src_seed[(sid, seed_id)].add(si)

    # resolved: (source_id, seed_id, sentence_idx) -> list[(qid, mention, ner)]
    linked_by_key: dict[tuple[str, str, int], list[tuple[str, str, str]]] = defaultdict(list)
    for r in resolved_rows:
        sid = str(r.get("source_id") or "")
        seed_id = str(r.get("seed_id") or "")
        si = int(r.get("sentence_idx", -1))
        if not sid or not seed_id or si < 0:
            continue
        qid = str(r.get("chosen_qid") or "").strip()
        men = str(r.get("mention") or "").strip()
        lab = str(r.get("ner_label") or "").strip()
        linked_by_key[(sid, seed_id, si)].append((qid, men, lab))

    seeds = load_seed_entities(root)
    seed_by_id = {s.seed_id: s for s in seeds}
    emap = load_entity_map(root / "sources" / "entity_map.csv")

    events_all: list[dict[str, Any]] = []
    n_groups = 0
    for (source_id, seed_id), idxs in sorted(idx_by_src_seed.items(), key=lambda x: (x[0][0], x[0][1])):
        sd = seed_by_id.get(seed_id)
        if not sd or not sd.qid or not sd.qid.startswith("Q"):
            continue
        idx_map = by_src_idx.get(source_id) or {}
        if not idx_map:
            continue
        seed_items = [(i, str(idx_map.get(i) or "").strip()) for i in sorted(idxs) if str(idx_map.get(i) or "").strip()]
        if not seed_items:
            continue
        meta = meta_by_source.get(source_id, {})
        url = meta.get("source_url", "")
        label = meta.get("source_label", source_id)
        cite_key = meta.get("citation_key", "")

        linked_by_sentence: dict[int, list[tuple[str, str, str]]] = {}
        for i, _s in seed_items:
            linked_by_sentence[int(i)] = list(linked_by_key.get((source_id, seed_id, int(i)), []))

        evs = extract_events_from_sentences(
            seed_items,
            seed_id=seed_id,
            seed_qid=sd.qid,
            source_id=source_id,
            source_url=url,
            source_label=label,
            citation_key=cite_key,
            entity_map=emap,
            linked_by_sentence=linked_by_sentence,
        )
        for e in evs:
            events_all.append(e.to_json())
        n_groups += 1

    out_path = curated / "events.jsonl"
    out_path.write_text("", encoding="utf-8")
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for r in events_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成：groups={n_groups} events={len(events_all)} 输出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

