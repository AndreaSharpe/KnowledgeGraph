"""从 data/*.csv 恢复 GraphBuild（用于 export_only）。"""

from __future__ import annotations

import csv
from pathlib import Path

from ..graph_model import GraphBuild


def load_graph_build_from_data_csv(data_dir: Path) -> GraphBuild:
    g = GraphBuild()
    nodes_path = data_dir / "nodes.csv"
    rels_path = data_dir / "relationships.csv"

    if nodes_path.is_file():
        with nodes_path.open(encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                nid = (row.get("id:ID") or row.get("id") or "").strip()
                if not nid:
                    continue
                labels_cell = (row.get(":LABEL") or "").strip()
                labels = [x.strip() for x in labels_cell.split(";") if x.strip()] if labels_cell else []
                g.ensure_node(
                    nid,
                    (row.get("name") or "").strip() or nid,
                    (row.get("extra") or "").strip() or None,
                    labels=labels,
                    props={k: v for k, v in row.items() if k not in ("id:ID", "id", "name", "extra", ":LABEL")},
                )

    if rels_path.is_file():
        with rels_path.open(encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                a = (row.get(":START_ID") or "").strip()
                b = (row.get(":END_ID") or "").strip()
                pid = (row.get("prop_id") or "").strip()
                plab = (row.get("prop_label") or pid).strip()
                dire = (row.get("direction") or "OUT").strip() or "OUT"
                if not a or not b:
                    continue
                g.add_edge(
                    a,
                    b,
                    pid,
                    plab,
                    dire,
                    provenance=(row.get("provenance") or "wikidata").strip() or "wikidata",
                    citation_key=(row.get("citation_key") or "").strip(),
                    snippet=(row.get("snippet") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                    layer=(row.get("layer") or "").strip() or None,
                    **{k: v for k, v in row.items() if k not in (":START_ID", ":END_ID", ":TYPE", "prop_id", "prop_label", "direction", "provenance", "layer", "citation_key", "snippet", "source_url")},
                )
    return g
