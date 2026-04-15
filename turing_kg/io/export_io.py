"""步骤四：导出 CSV / JSON / 抽取三元组表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import ROOT_ENTITY_QID
from ..graph_model import GraphBuild


def write_graph_csv_json(
    g: GraphBuild,
    out_dir: Path,
    *,
    bibliography: dict[str, dict] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.csv"
    rels_path = out_dir / "relationships.csv"

    with nodes_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id:ID", "name", "extra", ":LABEL"])
        w.writeheader()
        for n in g.nodes.values():
            w.writerow(
                {
                    "id:ID": n["id"],
                    "name": n["name"],
                    "extra": n.get("extra", ""),
                    ":LABEL": "Entity",
                }
            )

    with rels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                ":START_ID",
                ":END_ID",
                ":TYPE",
                "prop_id",
                "prop_label",
                "direction",
                "provenance",
                "citation_key",
                "snippet",
                "source_url",
            ],
        )
        w.writeheader()
        for e in g.edges:
            w.writerow(
                {
                    ":START_ID": e["start_id"],
                    ":END_ID": e["end_id"],
                    ":TYPE": e["rel_type"],
                    "prop_id": e["prop_id"],
                    "prop_label": e["prop_label"],
                    "direction": e["direction"],
                    "provenance": e.get("provenance", "wikidata"),
                    "citation_key": e.get("citation_key", ""),
                    "snippet": (e.get("snippet") or "").replace("\n", " ")[:500],
                    "source_url": e.get("source_url", ""),
                }
            )

    summary = {
        "root": ROOT_ENTITY_QID,
        "node_count": len(g.nodes),
        "edge_count": len(g.edges),
        "nodes": list(g.nodes.values()),
        "edges": g.edges,
        "bibliography": bibliography or {},
    }
    (out_dir / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_triples_csv(path: Path) -> list[dict[str, str]]:
    """读取与 write_triples_csv 兼容的三元组 CSV（无文件则返回 []）。"""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def write_triples_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "subject_qid,seed_id,seed_qid,sentence_idx,relevance_score,reasons,evidence_sentence,predicate,predicate_prop_id,object_mention,object_qid,object_is_literal,ner_label,link_score,citation_key,source_url,snippet,extraction_method\n",
            encoding="utf-8",
        )
        return
    # 导出采用“固定优先字段 + 兼容扩展字段”的策略：
    # - 新字段用于多 seed 归因与审计
    # - 旧调用方若没提供这些字段也能正常写出（默认空值）
    base = [
        "subject_qid",
        "seed_id",
        "seed_qid",
        "sentence_idx",
        "relevance_score",
        "reasons",
        "evidence_sentence",
        "predicate",
        "predicate_prop_id",
        "object_mention",
        "object_qid",
        "object_is_literal",
        "ner_label",
        "link_score",
        "citation_key",
        "source_url",
        "snippet",
        "extraction_method",
    ]
    extras: list[str] = []
    seen = set(base)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                extras.append(k)
                seen.add(k)
    fieldnames = base + extras
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
