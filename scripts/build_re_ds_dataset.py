#!/usr/bin/env python3
"""构建经典 DS 多分类数据集（含 NA）：re_ds_dataset_{seed_type}.jsonl。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from turing_kg.relation.ds_dataset import build_and_write_multiclass_dataset

    ap = argparse.ArgumentParser(description="构建 DS 多分类数据集（含 NA）")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--na-ratio", type=float, default=1.0, help="NA:POS 比例（默认 1.0）")
    ap.add_argument("--max-na-when-no-pos", type=int, default=200, help="当 POS=0 时仍保留的 NA 行数（用于诊断/标注）")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--p-train", type=float, default=0.8)
    ap.add_argument("--p-dev", type=float, default=0.1)
    ap.add_argument("--keep-multi-pos", action="store_true", help="若开启：labels_pos>1 的 bag 取第一个（不丢弃）")
    args = ap.parse_args()

    root = args.project_root.resolve()
    drop_multi = not bool(args.keep_multi_pos)
    path, n = build_and_write_multiclass_dataset(
        root,
        seed_type=args.seed_type,
        na_ratio=args.na_ratio,
        max_na_when_no_pos=args.max_na_when_no_pos,
        split_seed=args.split_seed,
        p_train=args.p_train,
        p_dev=args.p_dev,
        drop_multi_pos=drop_multi,
    )
    print(f"已写入 {path}，共 {n} 条 bag。")


if __name__ == "__main__":
    main()

