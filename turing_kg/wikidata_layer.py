"""步骤一：从 Wikidata 拉取中心实体的结构化陈述（API），写入图。"""

from __future__ import annotations

from .config import ROOT_ENTITY_QID
from .graph_model import GraphBuild
from .wikidata_api import (
    iter_claim_literal_snippets,
    load_root_neighborhood,
    pick_label,
)

#导入Wikidata数据包
def ingest_wikidata_bundle(g: GraphBuild, bundle: dict) -> None:
    root_qid = bundle["root_qid"]
    entities = bundle["entities"]
    edges = bundle["item_edges"]
    claims = bundle["claims"]

    root_label = pick_label(entities.get(root_qid, {"id": root_qid}))
    g.ensure_node(root_qid, root_label or "\u827e\u4f26\u00b7\u56fe\u7075")

    literal_props = {"P569", "P570"}
    parts: list[str] = []
    for pid, text in iter_claim_literal_snippets(claims, only_props=literal_props):
        plab = pick_label(entities.get(pid, {"id": pid}))
        parts.append(f"{plab}: {text}")
    if parts:
        g.ensure_node(
            root_qid,
            root_label,
            extra=" | ".join(sorted(set(parts))),
        )

    for prop_id, target_q in edges:
        prop_ent = entities.get(prop_id, {"id": prop_id})
        tgt_ent = entities.get(target_q, {"id": target_q})
        plabel = pick_label(prop_ent)
        tlabel = pick_label(tgt_ent)
        g.ensure_node(root_qid, root_label)
        g.ensure_node(target_q, tlabel or target_q)
        g.add_edge(root_qid, target_q, prop_id, plabel or prop_id, "OUT")

#加载结构化图
def load_structured_graph(root_qid: str = ROOT_ENTITY_QID) -> GraphBuild:
    g = GraphBuild()
    bundle = load_root_neighborhood(root_qid)
    ingest_wikidata_bundle(g, bundle)
    return g
