"""将 re_predictions.jsonl 转为与 build 一致的三元组行，并可写入 GraphBuild。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..attribution.triple_merge import merge_triple_rows
from ..graph_model import GraphBuild
from ..io.export_io import read_triples_csv
from .config_loaders import load_relation_thresholds, min_non_na_prob_for_seed_type, prop_label_for, threshold_for_prop
from .config_loaders import load_relation_allowlist, load_relation_schema
from .ds_labels import read_jsonl
from ..structured.wikidata_api import wbgetentities


def _graph_has_edge(g: GraphBuild, src: str, dst: str, prop_id: str, *, provenance: str | None = None) -> bool:
    return g.has_edge(src, dst, prop_id, provenance=provenance)


def _node_kinds(g: GraphBuild, qid: str) -> set[str]:
    n = g.nodes.get(qid) or {}
    labs = n.get("labels") or []
    return {str(x) for x in labs if x and x != "Entity"}


def _allowlist_ok(project_root: Path, *, seed_type: str, prop_id: str) -> bool:
    allow = load_relation_allowlist(project_root)
    by_type = allow.get("by_seed_type") or {}
    space = set(str(x) for x in (by_type.get(seed_type) or []))
    return prop_id in space


def _domain_range_ok(g: GraphBuild, *, seed_type: str, prop_id: str, object_qid: str) -> bool:
    """
    经典硬约束（最小可行，专治语义错配）：
    - 根据 prop_id 约束 object 的粗类型（labels）。
    - 只做强约束，不做“兜底放行”，宁可少边也不产错边。
    """
    obj_kinds = _node_kinds(g, object_qid)
    if prop_id in ("P19", "P20", "P27", "P551", "P17", "P276"):
        return "Location" in obj_kinds
    if prop_id in ("P69", "P108", "P1416", "P127", "P137", "P361"):
        return "Organization" in obj_kinds
    if prop_id in ("P166",):
        return "Award" in obj_kinds
    if prop_id in ("P279", "P31"):
        return "Concept" in obj_kinds
    if prop_id in ("P800",):
        # notable work：仅保留指向概念/作品类（目前没有 Work 类型时先用 Concept 近似）
        return "Concept" in obj_kinds
    return True


def _text_trigger_ok(project_root: Path, *, prop_id: str, evidence_sentence: str) -> bool:
    """
    经典触发词/句式约束（最小可行）：
    - 从 sources/relation_schema.json 读取 aliases_zh，要求句子命中至少一个触发词
    - 加入少量互斥规则，避免 P108(雇主) 与 P69(就读) 等常见错配
    - 若 schema 没配置 aliases_zh，则不做该约束（返回 True）
    """
    s = (evidence_sentence or "").strip()
    if not s:
        return False

    # 互斥规则（强）：学习类句子不允许映射到雇主/隶属等
    if prop_id in ("P108", "P1416"):
        if any(x in s for x in ("学习", "就读", "毕业", "入学", "在校", "攻读")):
            return False
    # 国籍/国家/地点：必须有地理触发（否则很容易误配）
    if prop_id in ("P27", "P17", "P276", "P19", "P20", "P551"):
        if not any(x in s for x in ("国籍", "公民", "国家", "位于", "在", "来自", "出生", "生于", "逝世", "去世", "定居", "居住", "地点")):
            return False

    data = load_relation_schema(project_root)
    rels = data.get("relations") or []
    aliases: list[str] = []
    for r in rels:
        if str(r.get("prop_id") or "") == prop_id:
            aliases = list(r.get("aliases_zh") or [])
            break
    aliases = [str(a).strip() for a in aliases if isinstance(a, str) and a.strip()]
    if not aliases:
        return True

    def _alias_match(alias: str, sent: str) -> bool:
        # 支持 “…” 槽位：要求分段按顺序出现（经典模板匹配）
        if "…" in alias:
            parts = [p.strip() for p in alias.split("…") if p.strip()]
            if not parts:
                return False
            idx = 0
            for p in parts:
                j = sent.find(p, idx)
                if j < 0:
                    return False
                idx = j + len(p)
            return True
        return alias in sent

    return any(_alias_match(a, s) for a in aliases if a)


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
    s = set(p31_qids)
    if "Q5" in s:
        return "Person"
    if s & {"Q618779", "Q11448906", "Q19020"}:
        return "Award"
    if s & {"Q43229", "Q79913", "Q484652"}:
        return "Organization"
    if s & {"Q17334923", "Q2221906", "Q515", "Q6256", "Q82794"}:
        return "Location"
    if s & {"Q151885", "Q7184903", "Q33104279"}:
        return "Concept"
    return ""


def _ensure_object_types(g: GraphBuild, object_qids: set[str]) -> None:
    """
    为 text_re 候选的 object_qid 尽力补齐粗类型标签（P31 推断），让 domain/range 校验可用。
    """
    qids = sorted({q for q in object_qids if isinstance(q, str) and q.startswith("Q")})
    if not qids:
        return
    try:
        ents = wbgetentities(qids, props="claims")
    except Exception:
        return
    for qid in qids:
        ent = ents.get(qid) if isinstance(ents, dict) else None
        if not isinstance(ent, dict):
            continue
        kind = _infer_kind_from_instance_of(_claim_entity_ids(ent, "P31"))
        if kind:
            g.ensure_node(qid, labels=(kind,))


def triple_rows_from_re_predictions(
    project_root: Path,
    predictions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    仅保留通过阈值的预测。
    阈值一律按当前 sources/relation_thresholds.json 用 score 即时计算（不沿用 re_predictions
    里推理时的 passed_threshold），这样改阈值后只需重新 export，不必重跑 infer。
    """
    thr_cfg = load_relation_thresholds(project_root)
    out: list[dict[str, str]] = []
    for rec in predictions:
        subject_qid = str(rec.get("subject_qid") or "")
        seed_id = str(rec.get("seed_id") or "")
        seed_type = str(rec.get("seed_type") or "")
        bag_id = str(rec.get("bag_id") or "")
        source_url = str(rec.get("source_url") or "")
        citation_key = str(rec.get("citation_key") or "")
        object_qid = str(rec.get("object_qid") or "")
        model_tag = str(rec.get("model") or "pcnn_mil")

        # v2：经典 DS 多分类输出（predicted_label/predicted_prob）
        if "predicted_label" in rec:
            pred_label = str(rec.get("predicted_label") or "NA")
            pred_prob = float(rec.get("predicted_prob") or 0.0)
            if pred_label == "NA":
                continue
            tau = min_non_na_prob_for_seed_type(thr_cfg, seed_type)
            if pred_prob < tau:
                continue
            ev = rec.get("evidence") or {}
            ev_sentence = str(ev.get("top_sentence") or "")
            si = ev.get("top_sentence_idx")
            si_str = str(si) if si is not None and int(si) >= 0 else ""

            plab = prop_label_for(project_root, pred_label)
            out.append(
                {
                    "subject_qid": subject_qid,
                    "seed_id": seed_id,
                    "seed_qid": subject_qid,
                    "sentence_idx": si_str,
                    "relevance_score": f"{pred_prob:.4f}",
                    "reasons": json.dumps({"bag_id": bag_id, "mil": True, "tau": tau}, ensure_ascii=False)[:800],
                    "evidence_sentence": ev_sentence[:500],
                    "predicate": plab,
                    "predicate_prop_id": pred_label,
                    "object_mention": "",
                    "object_qid": object_qid,
                    "object_is_literal": "0",
                    "ner_label": "",
                    "link_score": f"{pred_prob:.4f}",
                    "citation_key": citation_key,
                    "source_url": source_url,
                    "snippet": f"[{model_tag}] {ev_sentence[:400]}",
                    "extraction_method": f"{model_tag}:{pred_label}",
                }
            )
            continue

        # v1：旧版多标签输出（predictions[*].prop_id/score）
        for pr in rec.get("predictions") or []:
            prop_id = str(pr.get("prop_id") or "")
            score = float(pr.get("score") or 0.0)
            tau = threshold_for_prop(thr_cfg, prop_id)
            if score < tau:
                continue
            ev = pr.get("evidence") or {}
            ev_sentence = str(ev.get("top_sentence") or "")
            si = ev.get("top_sentence_idx")
            si_str = str(si) if si is not None and int(si) >= 0 else ""

            plab = prop_label_for(project_root, prop_id)
            out.append(
                {
                    "subject_qid": subject_qid,
                    "seed_id": seed_id,
                    "seed_qid": subject_qid,
                    "sentence_idx": si_str,
                    "relevance_score": f"{score:.4f}",
                    "reasons": json.dumps({"bag_id": bag_id, "mil": True}, ensure_ascii=False)[:800],
                    "evidence_sentence": ev_sentence[:500],
                    "predicate": plab,
                    "predicate_prop_id": prop_id,
                    "object_mention": "",
                    "object_qid": object_qid,
                    "object_is_literal": "0",
                    "ner_label": "",
                    "link_score": f"{score:.4f}",
                    "citation_key": citation_key,
                    "source_url": source_url,
                    "snippet": f"[{model_tag}] {ev_sentence[:400]}",
                    "extraction_method": f"{model_tag}:{prop_id}",
                }
            )
    return out


def write_triples_mil_csv(project_root: Path, rows: list[dict[str, str]]) -> Path:
    path = project_root / "data" / "triples_mil_extracted.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "subject_qid,seed_id,seed_qid,sentence_idx,relevance_score,reasons,evidence_sentence,"
            "predicate,predicate_prop_id,object_mention,object_qid,object_is_literal,ner_label,"
            "link_score,citation_key,source_url,snippet,extraction_method\n",
            encoding="utf-8",
        )
        return path
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
        for k in r:
            if k not in seen:
                extras.append(k)
                seen.add(k)
    fieldnames = base + extras
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def ingest_mil_edges_to_graph(
    g: GraphBuild,
    project_root: Path,
    predictions: list[dict[str, Any]],
    *,
    skip_if_same_edge: bool = True,
) -> int:
    """将通过阈值的边写入 GraphBuild（evidence layer）；阈值同 triple 导出，读当前 JSON。"""
    thr_cfg = load_relation_thresholds(project_root)
    # 先把 object 的粗类型补齐（否则 domain/range 会把本应保留的边全过滤掉）
    try:
        obj_qids = {str(r.get("object_qid") or "").strip() for r in predictions or []}
        _ensure_object_types(g, obj_qids)
    except Exception:
        pass
    # 语义级聚合：同一 (s,o,prop) 只留一条 text_re 边，证据累积到 evidence_snippets 中。
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in predictions:
        subject_qid = str(rec.get("subject_qid") or "")
        object_qid = str(rec.get("object_qid") or "")
        seed_type = str(rec.get("seed_type") or "")
        seed_id = str(rec.get("seed_id") or "")
        bag_id = str(rec.get("bag_id") or "")
        source_url = str(rec.get("source_url") or "")
        citation_key = str(rec.get("citation_key") or "")
        model_tag = str(rec.get("model") or "")
        if "predicted_label" in rec:
            pred_label = str(rec.get("predicted_label") or "NA")
            pred_prob = float(rec.get("predicted_prob") or 0.0)
            if pred_label == "NA":
                continue
            tau = min_non_na_prob_for_seed_type(thr_cfg, seed_type)
            if pred_prob < tau:
                continue
            if not _allowlist_ok(project_root, seed_type=seed_type, prop_id=pred_label):
                continue
            if not _domain_range_ok(g, seed_type=seed_type, prop_id=pred_label, object_qid=object_qid):
                continue
            plab = prop_label_for(project_root, pred_label)
            ev = rec.get("evidence") or {}
            ev_sentence = str(ev.get("top_sentence") or "")
            if not _text_trigger_ok(project_root, prop_id=pred_label, evidence_sentence=ev_sentence):
                continue
            g.ensure_node(subject_qid)
            g.ensure_node(object_qid, object_qid)
            supporting = 1 if _graph_has_edge(g, subject_qid, object_qid, pred_label, provenance="wikidata") else 0
            k = (subject_qid, object_qid, pred_label)
            cur = agg.get(k)
            snippet = ev_sentence[:400]
            ev_key = f"{citation_key}|{source_url}|{snippet}"
            if cur is None or float(pred_prob) > float(cur.get("score") or 0.0):
                agg[k] = {
                    "subject_qid": subject_qid,
                    "object_qid": object_qid,
                    "prop_id": pred_label,
                    "prop_label": plab,
                    "direction": "OUT",
                    "provenance": "text_re",
                    "citation_key": citation_key,
                    "source_url": source_url,
                    "snippet": snippet,
                    "score": float(pred_prob),
                    "tau": float(tau),
                    "model": model_tag or "pcnn_ds_mc_selectatt",
                    "seed_type": seed_type,
                    "seed_id": seed_id,
                    "bag_id": bag_id,
                    "is_supporting_fact": supporting,
                    "evidence_snippets": {ev_key},
                }
            else:
                # 累积证据
                es = cur.get("evidence_snippets")
                if isinstance(es, set):
                    es.add(ev_key)
            continue

        for pr in rec.get("predictions") or []:
            prop_id = str(pr.get("prop_id") or "")
            score = float(pr.get("score") or 0.0)
            tau = threshold_for_prop(thr_cfg, prop_id)
            if score < tau:
                continue
            if not _allowlist_ok(project_root, seed_type=seed_type, prop_id=prop_id):
                continue
            if not _domain_range_ok(g, seed_type=seed_type, prop_id=prop_id, object_qid=object_qid):
                continue
            plab = prop_label_for(project_root, prop_id)
            ev = pr.get("evidence") or {}
            ev_sentence = str(ev.get("top_sentence") or "")
            if not _text_trigger_ok(project_root, prop_id=prop_id, evidence_sentence=ev_sentence):
                continue
            g.ensure_node(subject_qid)
            g.ensure_node(object_qid, object_qid)
            supporting = 1 if _graph_has_edge(g, subject_qid, object_qid, prop_id, provenance="wikidata") else 0
            k = (subject_qid, object_qid, prop_id)
            cur = agg.get(k)
            snippet = ev_sentence[:400]
            ev_key = f"{citation_key}|{source_url}|{snippet}"
            if cur is None or float(score) > float(cur.get("score") or 0.0):
                agg[k] = {
                    "subject_qid": subject_qid,
                    "object_qid": object_qid,
                    "prop_id": prop_id,
                    "prop_label": plab,
                    "direction": "OUT",
                    "provenance": "text_re",
                    "citation_key": citation_key,
                    "source_url": source_url,
                    "snippet": snippet,
                    "score": float(score),
                    "tau": float(tau),
                    "model": model_tag or "pcnn_mil_v1",
                    "seed_type": seed_type,
                    "seed_id": seed_id,
                    "bag_id": bag_id,
                    "is_supporting_fact": supporting,
                    "evidence_snippets": {ev_key},
                }
            else:
                es = cur.get("evidence_snippets")
                if isinstance(es, set):
                    es.add(ev_key)

    # 写回图：每个 (s,o,prop) 一条边
    n = 0
    for _k, rec2 in agg.items():
        evs = rec2.get("evidence_snippets")
        ev_list = sorted(evs)[:8] if isinstance(evs, set) else []
        g.add_edge(
            rec2["subject_qid"],
            rec2["object_qid"],
            rec2["prop_id"],
            rec2["prop_label"],
            "OUT",
            provenance="text_re",
            citation_key=str(rec2.get("citation_key") or ""),
            snippet=str(rec2.get("snippet") or ""),
            source_url=str(rec2.get("source_url") or ""),
            score=float(rec2.get("score") or 0.0),
            tau=float(rec2.get("tau") or 0.0),
            model=str(rec2.get("model") or ""),
            seed_type=str(rec2.get("seed_type") or ""),
            seed_id=str(rec2.get("seed_id") or ""),
            bag_id=str(rec2.get("bag_id") or ""),
            is_supporting_fact=int(rec2.get("is_supporting_fact") or 0),
            evidence_snippets=ev_list,
        )
        n += 1
    return n


def load_re_predictions(project_root: Path) -> list[dict[str, Any]]:
    curated = project_root / "data" / "curated"
    main = curated / "re_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    if main.is_file() and main.stat().st_size > 0:
        rows.extend(read_jsonl(main))
    else:
        for p in sorted(curated.glob("re_predictions_pcnn_*.jsonl")):
            rows.extend(read_jsonl(p))
    return rows


def export_mil_triples_from_file(project_root: Path) -> tuple[Path, int]:
    preds = load_re_predictions(project_root)
    rows = triple_rows_from_re_predictions(project_root, preds)
    p = write_triples_mil_csv(project_root, rows)
    return p, len(rows)


def merge_mil_triples_csv_into_rows(
    project_root: Path,
    base_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """若存在 data/triples_mil_extracted.csv 且有数据，与 base 合并去重。"""
    p = project_root / "data" / "triples_mil_extracted.csv"
    extra = read_triples_csv(p)
    if not extra:
        return base_rows
    return merge_triple_rows(base_rows + extra)


def apply_mil_to_export_if_present(project_root: Path, g: GraphBuild, triple_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """
    1）合并 MIL 三元组行到导出列表；
    2）若存在 re_predictions*.jsonl，将通过阈值的 MIL 边写入 g（与同 (s,o,prop) 已有边去重）。
    返回 (合并后的 triple_rows, 新增 MIL 边条数)。
    """
    merged = merge_mil_triples_csv_into_rows(project_root, triple_rows)
    preds = load_re_predictions(project_root)
    n_edge = 0
    if preds:
        n_edge = ingest_mil_edges_to_graph(g, project_root, preds, skip_if_same_edge=True)
    return merged, n_edge
