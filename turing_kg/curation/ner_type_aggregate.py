"""NER 弱类型聚合：将 mention-level 的 ner_label 汇总到 entity(QID) 节点属性。

设计原则：
- 不把 NER 当作实体强类型（不写入 :LABEL），只作为弱证据写入节点属性；
- 仅聚合 PER/ORG/LOC（数值类 DATE/TIME/MONEY/PERCENT 视为 literal，不进入实体弱类型）；
- 支持增量更新 GraphBuild 节点属性（votes/top）。
"""

from __future__ import annotations

from typing import Any

from ..graph_model import GraphBuild


_KEEP = {"PER", "ORG", "LOC"}
_TIE_BREAK = {"PER": 3, "ORG": 2, "LOC": 1}


def _pick_top(votes: dict[str, int]) -> str:
    if not votes:
        return ""
    best = None
    best_score = -1
    for lab, n in votes.items():
        if not lab or n <= 0:
            continue
        sc = int(n) * 10 + int(_TIE_BREAK.get(lab, 0))
        if sc > best_score:
            best_score = sc
            best = lab
    return best or ""


def update_node_ner_votes(
    g: GraphBuild,
    qid: str,
    ner_label: str,
    *,
    weight: int = 1,
) -> None:
    q = (qid or "").strip()
    if not q.startswith("Q"):
        return
    lab = (ner_label or "").strip().upper()
    if lab not in _KEEP:
        return
    w = int(weight) if int(weight) > 0 else 1

    node = g.nodes.get(q)
    votes: dict[str, int] = {}
    if isinstance(node, dict):
        cur = node.get("ner_label_votes")
        if isinstance(cur, dict):
            for k, v in cur.items():
                try:
                    kk = str(k).strip().upper()
                    vv = int(v)
                    if kk and vv > 0:
                        votes[kk] = vv
                except Exception:
                    continue
    votes[lab] = int(votes.get(lab, 0)) + w
    top = _pick_top(votes)
    g.ensure_node(q, props={"ner_label_votes": votes, "ner_label_top": top})


def bulk_update_from_resolved_rows(g: GraphBuild, rows: list[dict[str, Any]]) -> None:
    """从 resolved.jsonl 行列表批量聚合（只统计 chosen_qid 与 ner_label）。"""
    for r in rows:
        qid = str(r.get("chosen_qid") or "").strip()
        ner = str(r.get("ner_label") or "").strip()
        if qid.startswith("Q") and ner:
            update_node_ner_votes(g, qid, ner)

