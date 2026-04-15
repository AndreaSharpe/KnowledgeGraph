#!/usr/bin/env python3
"""从 re_predictions*.jsonl 导出 data/triples_mil_extracted.csv（.venv）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turing_kg.relation.mil_ingest import export_mil_triples_from_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.project_root.resolve()
    path, n = export_mil_triples_from_file(root)
    print(f"已写入 {path}，共 {n} 条三元组（通过阈值）。")


if __name__ == "__main__":
    main()
