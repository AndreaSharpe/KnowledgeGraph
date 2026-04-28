#!/usr/bin/env python3
"""
从 re_ds_dataset_{seed_type}.jsonl 抽样生成人工标注 gold 文件。

用法：
  .venv\\Scripts\\python.exe scripts\\sample_re_gold.py --seed-type Person --split dev --n 200

输出：
  data/gold/re_gold_{seed_type}_{split}.jsonl

每行包含：bag_id、label（待填）、以及方便人工判断的 top_sentence。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.ds_labels import read_jsonl  # noqa: E402


def _pick_top_sentence(row: dict[str, Any]) -> tuple[int, str]:
    inst = list(row.get("instances") or [])
    if not inst:
        return -1, ""
    # bag 构造时通常已按 sentence_idx 排序，这里取第一句做人工快速判断
    i0 = inst[0]
    return int(i0.get("sentence_idx", -1)), str(i0.get("sentence") or "")[:500]


def main() -> None:
    ap = argparse.ArgumentParser(description="抽样生成关系抽取 gold 标注文件")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--seed-type", type=str, required=True, choices=["Person", "Concept", "Award"])
    ap.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.project_root.resolve()
    ds_path = root / "data" / "curated" / f"re_ds_dataset_{args.seed_type}.jsonl"
    rows = [r for r in read_jsonl(ds_path) if str(r.get("split") or "") == args.split]
    if not rows:
        raise SystemExit(f"未找到 split={args.split} 的数据：{ds_path}")

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    rows = rows[: max(1, min(args.n, len(rows)))]

    out_dir = root / "data" / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"re_gold_{args.seed_type}_{args.split}.jsonl"

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            si, sent = _pick_top_sentence(r)
            f.write(
                json.dumps(
                    {
                        "bag_id": str(r.get("bag_id") or ""),
                        "seed_type": args.seed_type,
                        "subject_qid": str(r.get("subject_qid") or ""),
                        "object_qid": str(r.get("object_qid") or ""),
                        "sentence_idx": si,
                        "sentence": sent,
                        # 人工填写：\"NA\" 或某个 \"Pxxx\"
                        "label": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"已写入 {out_path}（{len(rows)} 条）。请编辑 label 字段完成标注。")


if __name__ == "__main__":
    main()

