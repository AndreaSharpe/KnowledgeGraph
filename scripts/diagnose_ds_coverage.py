#!/usr/bin/env python3
"""
诊断 DS 覆盖率：为什么 ds_labels 里只有少数 prop 有正例？

做法（以 subject_qid 为中心）：
- 统计 bags.jsonl 中出现的 object_qid 集合
- 拉取 Wikidata 中 subject_qid 的 (prop_id, object_qid) item-edges
- 在 allowlist 空间内，统计每个 prop_id 的“可命中对象数”与实际命中数

输出：
- data/analysis/re_ds_coverage_{seed_type}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.config_loaders import labels_space_for_seed, load_relation_allowlist  # noqa: E402
from turing_kg.relation.ds_labels import read_jsonl  # noqa: E402
from turing_kg.structured.wikidata_api import iter_claim_item_edges, wbgetentities  # noqa: E402


def _fetch_edges(subject_qid: str) -> list[tuple[str, str]]:
    ents = wbgetentities([subject_qid], props="claims")
    ent = ents.get(subject_qid) or {}
    claims = ent.get("claims") or {}
    return list(iter_claim_item_edges(claims))


def main() -> None:
    ap = argparse.ArgumentParser(description="诊断 DS 正例覆盖率（bags vs Wikidata claims）")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--subject-qid", type=str, default="", help="默认从 bags.jsonl 推断（该 seed_type 内唯一 subject）")
    args = ap.parse_args()

    root = args.project_root.resolve()
    curated = root / "data" / "curated"
    bags = read_jsonl(curated / "bags.jsonl")

    bags_st = [b for b in bags if str(b.get("seed_type") or "") == args.seed_type]
    if not bags_st:
        raise SystemExit(f"bags.jsonl 中没有 seed_type={args.seed_type} 的 bag")

    subject_set = {str(b.get("subject_qid") or "") for b in bags_st}
    subject_set = {q for q in subject_set if q.startswith("Q")}
    if args.subject_qid:
        subject_qid = args.subject_qid.strip()
    else:
        if len(subject_set) != 1:
            raise SystemExit(f"无法自动确定 subject_qid（发现 {len(subject_set)} 个）：{sorted(subject_set)[:5]}")
        subject_qid = next(iter(subject_set))

    allowlist = load_relation_allowlist(root)
    space = labels_space_for_seed(allowlist, seed_type=args.seed_type, seed_id="")
    space_set = set(space)
    if not space:
        raise SystemExit(f"allowlist 空：seed_type={args.seed_type}")

    object_qids = [str(b.get("object_qid") or "") for b in bags_st]
    object_qids = [q for q in object_qids if q.startswith("Q")]
    object_set = set(object_qids)

    # Wikidata 中该 subject 的 allowlist 边
    edges = _fetch_edges(subject_qid)
    edges_in_space = [(p, o) for (p, o) in edges if p in space_set and str(o).startswith("Q")]

    # prop -> all objects in wikidata
    wk_obj_by_prop: dict[str, set[str]] = defaultdict(set)
    for p, o in edges_in_space:
        wk_obj_by_prop[p].add(str(o))

    # 实际 bags 命中
    hit_by_prop: dict[str, set[str]] = defaultdict(set)
    for p, objs in wk_obj_by_prop.items():
        for o in objs:
            if o in object_set:
                hit_by_prop[p].add(o)

    out = {
        "seed_type": args.seed_type,
        "subject_qid": subject_qid,
        "bags": {
            "n_bags": len(bags_st),
            "n_unique_objects": len(object_set),
            "top_object_qid": Counter(object_qids).most_common(20),
        },
        "allowlist": space,
        "wikidata": {
            "n_edges_in_allowlist": len(edges_in_space),
            "by_prop_object_count": {p: len(objs) for p, objs in sorted(wk_obj_by_prop.items())},
        },
        "intersection": {
            "by_prop_hit_count": {p: len(objs) for p, objs in sorted(hit_by_prop.items())},
            "by_prop_hit_objects": {p: sorted(list(objs)) for p, objs in sorted(hit_by_prop.items())},
        },
        "explain": [
            "若某 prop 在 wikidata 里 object_count 很大但 hit_count=0，说明语料/EL 没把对应对象链接成 QID。",
            "若 wikidata 里该 prop object_count 本来就 0（或很小），说明 Q7251 本身在该属性下无声明或非 item 值。",
        ],
    }

    out_dir = root / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"re_ds_coverage_{args.seed_type}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()

