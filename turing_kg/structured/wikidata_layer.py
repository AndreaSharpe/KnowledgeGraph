"""结构化层：从 Wikidata 拉取 seed 的结构化陈述并写入图。"""

from __future__ import annotations

from ..config import ROOT_ENTITY_QID
from ..graph_model import GraphBuild
from .wikidata_api import iter_claim_literal_snippets, load_root_neighborhood, pick_label


def ingest_wikidata_bundle(g: GraphBuild, bundle: dict) -> None:
    root_qid = bundle["root_qid"]
    entities = bundle["entities"]
    edges = bundle["item_edges"]
    claims = bundle["claims"]

    root_label = pick_label(entities.get(root_qid, {"id": root_qid}))
    g.ensure_node(root_qid, root_label or "seed")

    # 仅抽取少量字面值属性作为节点 extra（可扩展）
    literal_props = {"P569", "P570"}  # birth/death
    parts: list[str] = []
    for pid, text in iter_claim_literal_snippets(claims, only_props=literal_props):
        plab = pick_label(entities.get(pid, {"id": pid}))
        parts.append(f"{plab}: {text}")
    if parts:
        g.ensure_node(root_qid, root_label, extra=" | ".join(sorted(set(parts))))

    for prop_id, target_q in edges:
        prop_ent = entities.get(prop_id, {"id": prop_id})
        tgt_ent = entities.get(target_q, {"id": target_q})
        plabel = pick_label(prop_ent)
        tlabel = pick_label(tgt_ent)
        g.ensure_node(root_qid, root_label)
        g.ensure_node(target_q, tlabel or target_q)
        g.add_edge(root_qid, target_q, prop_id, plabel or prop_id, "OUT")


def load_structured_graph(root_qid: str = ROOT_ENTITY_QID) -> GraphBuild:
    g = GraphBuild()
    bundle = load_root_neighborhood(root_qid)
    ingest_wikidata_bundle(g, bundle)
    return g


def load_structured_graph_for_seeds(seed_qids: list[str]) -> GraphBuild:
    """
    多 seed 结构化层：对每个 seed 分别拉取邻域并合并入图。
    若 seed_qids 为空则回退到 ROOT_ENTITY_QID。
    """
    g = GraphBuild()
    qids = [q for q in (seed_qids or []) if isinstance(q, str) and q.startswith("Q")]
    if not qids:
        qids = [ROOT_ENTITY_QID]
    for qid in qids:
        bundle = load_root_neighborhood(qid)
        ingest_wikidata_bundle(g, bundle)
    return g

