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


def _infer_labels_from_types(type_qids: list[str]) -> tuple[str, ...]:
    """
    从 Wikidata 的类型系统（P31 instance of / P279 subclass of）推断节点 labels（可多标签）。

    约束：
    - 只用于 Neo4j :LABEL（不新增属性字段）；
    - 规则保持可解释：命中某些“类型QID”就打对应细分 label；
    - 若能取到类型但未命中任何规则，则给出最小可用的 Concept 兜底（避免只剩 Entity）。
    """
    s = {q for q in (type_qids or []) if isinstance(q, str) and q.startswith("Q")}
    if not s:
        return ()

    labels: set[str] = set()

    def add(*lbs: str) -> None:
        for lb in lbs:
            if lb:
                labels.add(lb)

    # --- Person / Award ---
    if "Q5" in s:
        add("Person")

    if s & {"Q618779", "Q11448906", "Q19020", "Q20006438"}:
        add("Award")
        if "Q20006438" in s:
            add("Fellowship")
        if "Q19020" in s:
            add("Prize")

    # --- Organization (+细分) ---
    if s & {"Q43229", "Q484652", "Q79913", "Q327333", "Q96888669", "Q1194970", "Q891723", "Q1589009", "Q129492653"}:
        add("Organization")
    if "Q327333" in s:
        add("GovernmentAgency")
    if s & {"Q1194970", "Q891723"}:
        add("Company")
    if "Q1589009" in s:
        add("MediaCompany")
    if "Q96888669" in s:
        add("Publisher")
    if "Q129492653" in s:
        add("MilitaryOrganization")

    # education orgs
    if s & {"Q3918", "Q902104", "Q15936437"}:
        add("Organization", "University")
    if s & {"Q2418495", "Q269770", "Q1713379", "Q5155053", "Q9842"}:
        add("Organization", "School")

    # archives / libraries
    if "Q27032435" in s:  # academic archive
        add("Organization", "Archive")
    if "Q1574516" in s:  # photo library
        add("Organization", "Library")

    # --- Location (+细分) ---
    if s & {
        "Q17334923",
        "Q2221906",
        "Q515",
        "Q6256",
        "Q82794",
        "Q33742",
        "Q1288568",
        "Q34770",
        "Q3957",
        "Q2755753",
        "Q4919932",
        "Q157570",
        "Q7265244",
    }:
        add("Location")
    if "Q6256" in s or "Q1288568" in s:
        add("Country")
    if "Q515" in s or "Q3957" in s or "Q2755753" in s:
        add("City")
    if s & {"Q4919932", "Q7265244", "Q157570"}:
        add("Facility")

    # --- Work / Concept 细分 ---
    # Work（文献/作品）优先：scholarly article、report、edition/version 等
    if s & {"Q13442814", "Q10383930", "Q3331189", "Q10870555", "Q21114848", "Q41298"}:
        add("Work")
    if "Q13442814" in s:
        add("ScholarlyArticle", "Paper")
    if "Q10383930" in s:  # academic journal
        add("Journal")
    if "Q21114848" in s or "Q41298" in s:  # magazine / journal (periodical)
        add("Periodical")
    if "Q3331189" in s:  # edition/version of a work
        add("WorkVersion")
    if "Q10870555" in s:  # report
        add("Report")

    # Concept（抽象概念/类型/学科/名称等）
    if s & {"Q151885", "Q7184903", "Q33104279", "Q28640", "Q12737077", "Q11862829", "Q1047113", "Q483394"}:
        add("Concept")
    # profession / occupation / discipline family
    if "Q28640" in s:
        add("Profession")
    if "Q12737077" in s:
        add("Occupation")
    if s & {"Q11862829", "Q4671286"}:  # academic discipline / academic major
        add("Discipline")
    if "Q1047113" in s:  # specialty / field of study
        add("Field")

    # STEM / logic / math / CS (coarse domain tags)
    if s & {"Q7991"}:  # natural science
        add("Science")
    if s & {"Q901"}:  # mathematics
        add("Mathematics", "Science")
    if s & {"Q1936384"}:
        add("Mathematics")
    if s & {"Q123370638"}:
        add("ComputerScience")
    if s & {"Q816264"}:  # formal logic
        add("Logic")
    if "Q24034552" in s:
        add("Concept", "MathematicalConcept")
    if "Q1936384" in s:
        add("Concept", "Field", "MathematicsBranch")
    if "Q123370638" in s:
        add("Concept", "Field", "ComputerScienceBranch")
    if "Q112193867" in s:
        add("Concept", "DiseaseClass")
    if "Q31629" in s:
        add("Concept", "SportType")
    if "Q4830453" in s:
        add("Concept", "Business")

    # machines / automata / formal systems
    if s & {"Q137172521"}:  # non-finite-state machine
        add("Concept", "Machine", "Automaton")

    # names
    if "Q101352" in s:
        add("Concept", "Name", "FamilyName")
    if "Q12308941" in s:
        add("Concept", "Name", "GivenName")

    # Wikimedia / WikiProject navigation-ish（仍归为 Concept，避免混进 Organization/Work）
    if s & {"Q14204246", "Q16695773", "Q51539995", "Q10823887", "Q14827288", "Q33120876", "Q136375263"}:
        add("Concept", "Wikimedia")
    if "Q16695773" in s or "Q51539995" in s:
        add("WikiProject")

    # 兜底：有类型但没命中任何规则 → Concept
    if not labels:
        add("Concept")

    # 规范：子类 label 命中时补齐父类 label
    if labels & {"University", "School", "Company", "Publisher", "GovernmentAgency", "Archive", "Library", "MediaCompany", "MilitaryOrganization"}:
        add("Organization")
    if labels & {"Country", "City", "Facility"}:
        add("Location")
    if labels & {"ScholarlyArticle", "Paper", "Journal", "Periodical", "WorkVersion", "Report"}:
        add("Work")

    return tuple(sorted(labels))


_TOP_KIND_LABELS: tuple[str, ...] = ("Person", "Award", "Concept", "Organization", "Location", "Work")


def enrich_kind_labels_for_graph(g: GraphBuild, *, batch_size: int = 50) -> None:
    """
    为图中缺少细粒度类型标签的 Q 节点补齐 :Person/:Award/:Concept/:Organization/:Location。

    规则：
    - 仅处理 id 以 Q 开头的节点；
    - 若节点已有上述任一 kind label，则跳过；
    - 否则尝试从 Wikidata 取该 Q 的 claims(P31) 推断 kind，并补标签；
    - 不影响 Event/Literal 等非 Q 节点。
    """
    qids: list[str] = []
    for nid, n in g.nodes.items():
        if not isinstance(nid, str) or not nid.startswith("Q"):
            continue
        labs = list((n or {}).get("labels") or [])
        if any(lb in _TOP_KIND_LABELS for lb in labs):
            continue
        qids.append(nid)
    if not qids:
        return

    # 分批取 claims，避免一次请求过大/触发限流
    for i in range(0, len(qids), max(1, int(batch_size))):
        chunk = qids[i : i + max(1, int(batch_size))]
        try:
            ents = wbgetentities(chunk, props="claims", languages="zh-hans|zh|en")
        except Exception:
            continue
        for qid in chunk:
            ent = ents.get(qid) if isinstance(ents, dict) else None
            if not isinstance(ent, dict):
                continue
            p31 = _claim_entity_ids(ent, "P31")
            p279 = _claim_entity_ids(ent, "P279")
            lbs = _infer_labels_from_types(p31 + p279)
            if lbs:
                g.ensure_node(qid, labels=lbs)


def ingest_wikidata_bundle(g: GraphBuild, bundle: dict) -> None:
    root_qid = bundle["root_qid"]
    entities = bundle["entities"]
    edges = bundle["item_edges"]
    claims = bundle["claims"]

    root_label = pick_label(entities.get(root_qid, {"id": root_qid}))
    root_ent = entities.get(root_qid, {"id": root_qid})
    root_types = _claim_entity_ids(root_ent, "P31") + _claim_entity_ids(root_ent, "P279") if isinstance(root_ent, dict) else []
    root_labels = _infer_labels_from_types(root_types)
    g.ensure_node(root_qid, root_label or "seed", labels=root_labels)

    # 仅抽取少量字面值属性作为节点 extra（可扩展）
    literal_props = {"P569", "P570"}  # birth/death
    parts: list[str] = []
    for pid, text in iter_claim_literal_snippets(claims, only_props=literal_props):
        plab = pick_label(entities.get(pid, {"id": pid}))
        parts.append(f"{plab}: {text}")
    if parts:
        g.ensure_node(root_qid, root_label, extra=" | ".join(sorted(set(parts))), labels=root_labels)

    for prop_id, target_q in edges:
        if prop_id in _DROP_PROPS:
            continue
        prop_ent = entities.get(prop_id, {"id": prop_id})
        tgt_ent = entities.get(target_q, {"id": target_q})
        plabel = _PROP_LABEL_OVERRIDES_ZH.get(prop_id) or pick_label(prop_ent)
        tlabel = pick_label(tgt_ent)
        g.ensure_node(root_qid, root_label)
        tgt_types = _claim_entity_ids(tgt_ent, "P31") + _claim_entity_ids(tgt_ent, "P279") if isinstance(tgt_ent, dict) else []
        tgt_labels = _infer_labels_from_types(tgt_types)
        g.ensure_node(target_q, tlabel or target_q, labels=tgt_labels)
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
    ftypes = _claim_entity_ids(focal_ent, "P31") + _claim_entity_ids(focal_ent, "P279") if isinstance(focal_ent, dict) else []
    flabels = _infer_labels_from_types(ftypes)
    g.ensure_node(object_qid, focal_label or object_qid, labels=flabels)

    for src, prop_id, dst in triples:
        if prop_id in _DROP_PROPS:
            continue
        if g.has_edge(src, dst, prop_id):
            continue
        se = ents.get(src, {"id": src})
        pe = ents.get(prop_id, {"id": prop_id})
        slabel = pick_label(se) if isinstance(se, dict) else src
        plab = _PROP_LABEL_OVERRIDES_ZH.get(prop_id) or (pick_label(pe) if isinstance(pe, dict) else prop_id)
        stypes = _claim_entity_ids(se, "P31") + _claim_entity_ids(se, "P279") if isinstance(se, dict) else []
        slabels = _infer_labels_from_types(stypes)
        g.ensure_node(src, slabel or src, labels=slabels)
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

