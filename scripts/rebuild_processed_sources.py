#!/usr/bin/env python3
"""
只重建 processed 层：sentences.jsonl + routing.jsonl。

用途：
- 新增/更新 raw/excerpts/articles/*.txt 后，不想跑 run.py --mode full，
  可用此脚本把新语料快速纳入 processed，再配合 rebuild_curated_ner_el.py 重建 NER/EL 中间层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="重建 processed/sentences.jsonl + routing.jsonl（不跑 NER/EL）")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.project_root.resolve()

    from turing_kg.attribution.seed_config import load_attribution_config, load_seed_entities
    from turing_kg.attribution.seed_router import route_sentences
    from turing_kg.extraction.ner_link import _split_sentences, _zh_ratio
    from turing_kg.sources.text_sources import collect_text_sources
    from turing_kg.io.stage_io import write_processed_sentences_jsonl, write_routing_jsonl

    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "sentences.jsonl").write_text("", encoding="utf-8")
    (processed / "routing.jsonl").write_text("", encoding="utf-8")

    seeds = load_seed_entities(root)
    at_cfg = load_attribution_config(root)

    n_sources = 0
    n_sent = 0
    n_routed = 0
    for ch, _prov, cite_key, url in collect_text_sources(root):
        label = f"{ch.lang}:{ch.title}"
        source_id = url or label
        use_zh = _zh_ratio(ch.text) >= 0.12
        sentences = _split_sentences(ch.text, use_zh)
        if not sentences:
            continue
        write_processed_sentences_jsonl(
            root,
            source_id=source_id,
            source_url=url,
            source_label=label,
            citation_key=cite_key,
            sentences=sentences,
        )
        n_sent += len(sentences)
        routed = route_sentences(sentences, seeds=seeds, cfg=at_cfg)
        # 与 build.py 保持一致：若整篇没有任何归因，则兜底把全部句子归到 turing_person（保证下游可运行）。
        fallback = next((s for s in seeds if s.seed_id == "turing_person"), None)
        need_fallback = (fallback is not None) and (not any(ra.assigned for ra in routed))
        audit_rows = []
        for i, ra in enumerate(routed):
            assigned = []
            if ra.assigned:
                for a in ra.assigned:
                    assigned.append(
                        {
                            "seed_id": a.seed_id,
                            "seed_qid": a.seed_qid,
                            "score": float(a.score),
                            "reasons": a.reasons,
                        }
                    )
            elif need_fallback and fallback is not None:
                assigned.append(
                    {
                        "seed_id": fallback.seed_id,
                        "seed_qid": fallback.qid,
                        "score": 1.0,
                        "reasons": {"fallback_all_sentences": True},
                    }
                )
            audit_rows.append({"sentence_idx": i, "sentence": ra.sentence, "assigned": assigned})
        write_routing_jsonl(
            root,
            source_id=source_id,
            source_url=url,
            source_label=label,
            citation_key=cite_key,
            routed_rows=audit_rows,
        )
        n_routed += len(audit_rows)
        n_sources += 1

    print(f"完成：sources={n_sources} sentences={n_sent} routing_rows={n_routed}")


if __name__ == "__main__":
    main()

