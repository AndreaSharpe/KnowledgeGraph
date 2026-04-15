from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reset_stage_files(project_root: Path) -> None:
    """
    清空中间层文件（避免多次运行累积造成不可复现）。
    """
    for rel in (
        Path("data/processed/sentences.jsonl"),
        Path("data/processed/routing.jsonl"),
        Path("data/curated/mentions.jsonl"),
        Path("data/curated/candidates.jsonl"),
        Path("data/curated/resolved.jsonl"),
        Path("data/curated/bags.jsonl"),
        Path("data/curated/ds_labels.jsonl"),
        Path("data/curated/re_predictions.jsonl"),
    ):
        p = project_root / rel
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
        except Exception:
            pass


def write_processed_sentences_jsonl(
    project_root: Path,
    *,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    sentences: list[str],
) -> None:
    path = project_root / "data" / "processed" / "sentences.jsonl"
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(sentences):
        rows.append(
            {
                "source_id": source_id,
                "source_url": source_url,
                "source_label": source_label,
                "citation_key": citation_key,
                "paragraph_idx": 0,
                "sentence_idx": i,
                "sentence": s,
            }
        )
    _append_jsonl(path, rows)


def write_routing_jsonl(
    project_root: Path,
    *,
    source_id: str,
    source_url: str,
    source_label: str,
    citation_key: str,
    routed_rows: list[dict[str, Any]],
) -> None:
    """
    每条 routed_rows 建议至少含：
    - sentence_idx, sentence
    - assigned: [{seed_id, seed_qid, score, reasons}]
    """
    path = project_root / "data" / "processed" / "routing.jsonl"
    rows: list[dict[str, Any]] = []
    for r in routed_rows:
        rows.append(
            {
                "source_id": source_id,
                "source_url": source_url,
                "source_label": source_label,
                "citation_key": citation_key,
                **r,
            }
        )
    _append_jsonl(path, rows)


def write_mentions_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> None:
    _append_jsonl(project_root / "data" / "curated" / "mentions.jsonl", rows)


def write_candidates_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> None:
    _append_jsonl(project_root / "data" / "curated" / "candidates.jsonl", rows)


def write_resolved_jsonl(project_root: Path, rows: list[dict[str, Any]]) -> None:
    _append_jsonl(project_root / "data" / "curated" / "resolved.jsonl", rows)

