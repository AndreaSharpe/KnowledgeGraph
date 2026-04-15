from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sentence_attribution_jsonl(
    path: Path,
    *,
    source_url: str,
    source_label: str,
    rows: list[dict[str, Any]],
) -> None:
    """
    逐句归因审计落盘（JSONL）。

    rows 每条建议包含：
    - sentence_idx
    - sentence
    - assigned: [{seed_id, seed_qid, score, reasons}]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "source_url": source_url,
                "source_label": source_label,
                **r,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

