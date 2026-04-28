"""结构化层：从 Wikidata 拉取 seed 的结构化陈述并写入图。"""

from __future__ import annotations

from typing import Any

from ..config import ROOT_ENTITY_QID
from ..graph_model import GraphBuild
from .wikidata_api import (
    fetch_incoming_item_edges,
    iter_claim_literal_snippets,
    load_root_neighborhood,
    pick_label,
    wbgetentities,
)


_PROP_LABEL_OVERRIDES_ZH: dict[str, str] = {
    "P31": "实例",
    "P279": "子类",
    "P361": "隶属于",
    "P127": "所有者",
    "P137": "运营方",
    "P17": "国家",
    "P276": "地点",
    "P800": "主要作品",
    "P166": "所获奖项",
    "P921": "作品主题",
    "P1343": "记载于",
    "P366": "有用途",
    "P1535": "使用者",
    "P2579": "研究学科",
    "P1344": "参与者",
}

# 对「图灵机 / 图灵奖」补充 Wikidata 入向边（?主体 wdt:P* ?焦点）；条数用 LIMIT 控制，避免图爆炸
_FOCAL_INCOMING_WD: dict[str, tuple[tuple[str, ...], int]] = {
    "Q163310": (
        ("P800", "P921", "P1343", "P361", "P366", "P2579", "P1535"),
        50,
    ),
    "Q185667": (("P166", "P1343", "P921", "P361", "P1344"), 45),
}

# 不进入最终领域图谱的“百科维护/导航”属性（会引入 Category/Template 等噪声）
_DROP_PROPS = {"P910", "P1424"}


def _claim_entity_ids(ent: dict[str, Any], prop_id: str) -> list[str]:
    claims = ent.get("claims") or {}
    statements = claims.get(prop_id) or []
    out: list[str] = []
    if not isinstance(statements, list):
        return out
    for st in statements:
        snak = (st or {}).get("mainsnak") or {}
        if snak.get("snaktype") != "value":
            continue
        dv = snak.get("datavalue") or {}
        if dv.get("type") != "wikibase-entityid":
            continue
        tid = (dv.get("value") or {}).get("id")
        if isinstance(tid, str) and tid.startswith("Q"):
            out.append(tid)
    return out


def _infer_kind_from_instance_of(p31_qids: list[str]) -> str:
    """
    从 P31(instance of) 做粗类型推断（最小可行，用于 Neo4j :LABEL）。
    只要命中即返回；否则返回空字符串。
    """
    s = set(p31_qids)
    # human
    if "Q5" in s:
        return "Person"
    # award / prize
    if s & {"Q618779", "Q11448906", "Q19020"}:  # award / science award / prize
        return "Award"
    # organization-ish
    if s & {"Q43229", "Q79913", "Q484652"}:  # organization / educational institution / international organization
        return "Organization"
    # location-ish
    if s & {"Q17334923", "Q2221906", "Q515", "Q6256", "Q82794"}:  # location / geographic region / city / country / geographic region
        return "Location"
    # concept-ish (fallback for abstract types)
    if s & {"Q151885", "Q7184903", "Q33104279"}:  # concept / abstract object / scientific concept
        return "Concept"
    return ""


def ingest_wikidata_bundle(g: GraphBuild, bundle: dict) -> None:
    root_qid = bundle["root_qid"]
    entities = bundle["entities"]
    edges = bundle["item_edges"]
    claims = bundle["claims"]

    root_label = pick_label(entities.get(root_qid, {"id": root_qid}))
    root_ent = entities.get(root_qid, {"id": root_qid})
    root_kind = _infer_kind_from_instance_of(_claim_entity_ids(root_ent, "P31")) if isinstance(root_ent, dict) else ""
    g.ensure_node(root_qid, root_label or "seed", labels=(root_kind,) if root_kind else ())

    # 仅抽取少量字面值属性作为节点 extra（可扩展）
    literal_props = {"P569", "P570"}  # birth/death
    parts: list[str] = []
    for pid, text in iter_claim_literal_snippets(claims, only_props=literal_props):
        plab = pick_label(entities.get(pid, {"id": pid}))
        parts.append(f"{plab}: {text}")
    if parts:
        g.ensure_node(root_qid, root_label, extra=" | ".join(sorted(set(parts))), labels=(root_kind,) if root_kind else ())

    for prop_id, target_q in edges:
        if prop_id in _DROP_PROPS:
            continue
        prop_ent = entities.get(prop_id, {"id": prop_id})
        tgt_ent = entities.get(target_q, {"id": target_q})
        plabel = _PROP_LABEL_OVERRIDES_ZH.get(prop_id) or pick_label(prop_ent)
        tlabel = pick_label(tgt_ent)
        g.ensure_node(root_qid, root_label)
        kind = _infer_kind_from_instance_of(_claim_entity_ids(tgt_ent, "P31")) if isinstance(tgt_ent, dict) else ""
        g.ensure_node(target_q, tlabel or target_q, labels=(kind,) if kind else ())
        g.add_edge(root_qid, target_q, prop_id, plabel or prop_id, "OUT")


def _ingest_incoming_wikidata_edges(
    g: GraphBuild,
    object_qid: str,
    *,
    allowed_props: tuple[str, ...],
    limit: int,
) -> None:
    """
    将「其它条目 -> object_qid」的 Wikidata 直接边写入图（与 load_root 的出边方向一致：start 为主体，end 为宾语 Q）。
    与已存在的同 (src,dst,prop) 边去重；provenance 用 wikidata_incoming 便于与「根条目拉邻域」区分。
    """
    triples = fetch_incoming_item_edges(object_qid, allowed_props, limit=limit)
    if not triples:
        return
    subj_ids = {t[0] for t in triples}
    prop_ids = {t[1] for t in triples}
    want: list[str] = sorted({object_qid} | subj_ids | prop_ids)
    ents: dict[str, dict] = {}
    try:
        for i in range(0, len(want), 20):
            chunk = want[i : i + 20]
            ents.update(wbgetentities(chunk, props="labels|claims") or {})
    except Exception:
        return

    focal_label = pick_label(ents.get(object_qid, {"id": object_qid}))
    focal_ent = ents.get(object_qid, {"id": object_qid})
    fkind = _infer_kind_from_instance_of(_claim_entity_ids(focal_ent, "P31")) if isinstance(focal_ent, dict) else ""
    g.ensure_node(object_qid, focal_label or object_qid, labels=(fkind,) if fkind else ())

    for src, prop_id, dst in triples:
        if prop_id in _DROP_PROPS:
            continue
        if g.has_edge(src, dst, prop_id):
            continue
        se = ents.get(src, {"id": src})
        pe = ents.get(prop_id, {"id": prop_id})
        slabel = pick_label(se) if isinstance(se, dict) else src
        plab = _PROP_LABEL_OVERRIDES_ZH.get(prop_id) or (pick_label(pe) if isinstance(pe, dict) else prop_id)
        skind = _infer_kind_from_instance_of(_claim_entity_ids(se, "P31")) if isinstance(se, dict) else ""
        g.ensure_node(src, slabel or src, labels=(skind,) if skind else ())
        g.ensure_node(dst, focal_label or dst)
        g.add_edge(
            src,
            dst,
            prop_id,
            plab or prop_id,
            "OUT",
            provenance="wikidata_incoming",
            wikidata_edge_mode="inlink_to_focal",
        )


def ingest_focal_incoming_bundles(g: GraphBuild, specs: dict[str, tuple[tuple[str, ...], int]] | None = None) -> None:
    """为配置的焦点 Q（图灵机、图灵奖等）批量追加 Wikidata 入边。"""
    s = specs if specs is not None else _FOCAL_INCOMING_WD
    for oq, (props, lim) in s.items():
        if not isinstance(oq, str) or not oq.startswith("Q"):
            continue
        _ingest_incoming_wikidata_edges(g, oq, allowed_props=props, limit=lim)


def load_structured_graph(root_qid: str = ROOT_ENTITY_QID) -> GraphBuild:
    g = GraphBuild()
    bundle = load_root_neighborhood(root_qid)
    ingest_wikidata_bundle(g, bundle)
    ingest_focal_incoming_bundles(g)
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
    ingest_focal_incoming_bundles(g)
    return g

