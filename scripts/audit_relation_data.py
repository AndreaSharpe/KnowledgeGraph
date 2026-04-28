#!/usr/bin/env python3
"""
关系抽取数据体检（DS/MIL）。

输出：
- data/analysis/re_stats.json：总体统计与按 seed_type/prop_id 的分布

仅做统计，不修改任何中间层文件。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.ds_labels import read_jsonl  # noqa: E402


def _safe_len(x: Any) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0


def main() -> None:
    ap = __import__("argparse").ArgumentParser(description="审计关系抽取 bags/ds_labels 统计分布")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    args = ap.parse_args()
    root: Path = args.project_root.resolve()

    curated = root / "data" / "curated"
    bags_path = curated / "bags.jsonl"
    ds_path = curated / "ds_labels.jsonl"
    preds_path = curated / "re_predictions.jsonl"

    bags = read_jsonl(bags_path)
    ds = read_jsonl(ds_path)
    preds = read_jsonl(preds_path) if preds_path.is_file() else []

    ds_by_bag = {str(r.get("bag_id") or ""): r for r in ds if str(r.get("bag_id") or "")}

    # 总体
    bag_sizes = []
    seed_type_counter = Counter()
    seed_id_counter = Counter()
    zh_filtered_note = "bags.jsonl 已包含 zh_ratio 门控后的实例；此脚本不再重复门控。"

    # DS 分布
    n_bags = 0
    n_with_ds_row = 0
    n_pos_any = 0
    pos_prop_counter = Counter()
    pos_prop_by_seed_type: dict[str, Counter[str]] = defaultdict(Counter)
    multi_pos_bags = 0
    no_pos_bags = 0

    # predictions 分布（可选）
    n_preds = len(preds)
    n_pred_has_any_passed = 0

    for b in bags:
        n_bags += 1
        seed_type = str(b.get("seed_type") or "Unknown")
        seed_id = str(b.get("seed_id") or "")
        seed_type_counter[seed_type] += 1
        if seed_id:
            seed_id_counter[seed_id] += 1
        inst = b.get("instances") or []
        bag_sizes.append(_safe_len(inst))

        bid = str(b.get("bag_id") or "")
        d = ds_by_bag.get(bid)
        if not d:
            continue
        n_with_ds_row += 1
        labels_pos = list(d.get("labels_pos") or [])
        if labels_pos:
            n_pos_any += 1
            if len(labels_pos) > 1:
                multi_pos_bags += 1
            for p in labels_pos:
                pos_prop_counter[p] += 1
                pos_prop_by_seed_type[seed_type][p] += 1
        else:
            no_pos_bags += 1

    for r in preds:
        any_passed = False
        for pr in r.get("predictions") or []:
            if bool(pr.get("passed_threshold")):
                any_passed = True
                break
        if any_passed:
            n_pred_has_any_passed += 1

    def _percent(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    bag_sizes_sorted = sorted(bag_sizes)
    p50 = bag_sizes_sorted[len(bag_sizes_sorted) // 2] if bag_sizes_sorted else 0
    p90 = bag_sizes_sorted[int(len(bag_sizes_sorted) * 0.9)] if bag_sizes_sorted else 0

    out = {
        "paths": {
            "bags": str(bags_path),
            "ds_labels": str(ds_path),
            "re_predictions": str(preds_path) if preds_path.is_file() else "",
        },
        "notes": [zh_filtered_note],
        "bags": {
            "n_bags": n_bags,
            "bag_size": {
                "min": int(min(bag_sizes)) if bag_sizes else 0,
                "p50": int(p50),
                "p90": int(p90),
                "max": int(max(bag_sizes)) if bag_sizes else 0,
                "avg": float(sum(bag_sizes) / len(bag_sizes)) if bag_sizes else 0.0,
            },
            "by_seed_type": dict(seed_type_counter),
            "top_seed_id": seed_id_counter.most_common(20),
        },
        "ds": {
            "n_ds_rows": n_with_ds_row,
            "coverage_of_bags": _percent(n_with_ds_row, n_bags),
            "n_pos_any": n_pos_any,
            "pos_any_rate": _percent(n_pos_any, n_with_ds_row),
            "n_no_pos": no_pos_bags,
            "no_pos_rate": _percent(no_pos_bags, n_with_ds_row),
            "n_multi_pos": multi_pos_bags,
            "multi_pos_rate": _percent(multi_pos_bags, n_with_ds_row),
            "pos_prop_counts": pos_prop_counter.most_common(),
            "pos_prop_counts_by_seed_type": {k: v.most_common() for k, v in pos_prop_by_seed_type.items()},
        },
        "predictions_snapshot": {
            "n_pred_rows": n_preds,
            "n_bags_with_any_passed_threshold": n_pred_has_any_passed,
            "passed_any_rate": _percent(n_pred_has_any_passed, n_preds),
        },
    }

    out_dir = root / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "re_stats.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()

