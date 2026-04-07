from __future__ import annotations

import csv
from pathlib import Path

from .graph_model import GraphBuild


def merge_manual_csv(g: GraphBuild, csv_path: Path) -> int:
    """
    读取人工补录 CSV，合并进图。
    列：start_id,end_id,prop_id,prop_label,direction,source（source 可选）
    prop_id 须为 Wikidata P 编号（如 P937），以便与 WIKI_Pxx 关系命名一致。
    """
    if not csv_path.is_file():
        return 0
    added = 0
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("start_id") or "").strip()
            eid = (row.get("end_id") or "").strip()
            pid = (row.get("prop_id") or "").strip()
            plab = (row.get("prop_label") or pid).strip()
            dire = (row.get("direction") or "OUT").strip().upper()
            if not sid or not eid or not pid or dire not in ("OUT", "IN"):
                continue
            if not pid.startswith("P") or not pid[1:].isdigit():
                continue
            g.ensure_node(sid, sid)
            g.ensure_node(eid, eid)
            if dire == "OUT":
                g.add_edge(sid, eid, pid, plab, "OUT")
            else:
                g.add_edge(sid, eid, pid, plab, "IN")
            added += 1
    return added
