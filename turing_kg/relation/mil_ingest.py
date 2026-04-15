"""将 re_predictions.jsonl 转为与 build 一致的三元组行，并可写入 GraphBuild。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..attribution.triple_merge import merge_triple_rows
from ..graph_model import GraphBuild
from ..io.export_io import read_triples_csv
from .config_loaders import load_relation_thresholds, prop_label_for, threshold_for_prop
from .ds_labels import read_jsonl


def _graph_has_edge(g: GraphBuild, src: str, dst: str, prop_id: str) -> bool:
    for e in g.edges:
        if e.get("start_id") == src and e.get("end_id") == dst and e.get("prop_id") == prop_id:
            return True
    return False


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
        bag_id = str(rec.get("bag_id") or "")
        source_url = str(rec.get("source_url") or "")
        citation_key = str(rec.get("citation_key") or "")
        object_qid = str(rec.get("object_qid") or "")
        model_tag = str(rec.get("model") or "pcnn_mil")

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
    """将通过阈值的边写入 GraphBuild（provenance=mil_relation_extraction）；阈值同 triple 导出，读当前 JSON。"""
    thr_cfg = load_relation_thresholds(project_root)
    n = 0
    for rec in predictions:
        subject_qid = str(rec.get("subject_qid") or "")
        object_qid = str(rec.get("object_qid") or "")
        source_url = str(rec.get("source_url") or "")
        citation_key = str(rec.get("citation_key") or "")
        for pr in rec.get("predictions") or []:
            prop_id = str(pr.get("prop_id") or "")
            score = float(pr.get("score") or 0.0)
            tau = threshold_for_prop(thr_cfg, prop_id)
            if score < tau:
                continue
            if skip_if_same_edge and _graph_has_edge(g, subject_qid, object_qid, prop_id):
                continue
            plab = prop_label_for(project_root, prop_id)
            ev = pr.get("evidence") or {}
            ev_sentence = str(ev.get("top_sentence") or "")
            g.ensure_node(subject_qid)
            g.ensure_node(object_qid, object_qid)
            g.add_edge(
                subject_qid,
                object_qid,
                prop_id,
                plab,
                "OUT",
                provenance="mil_relation_extraction",
                citation_key=citation_key,
                snippet=f"{ev_sentence[:400]} | pcnn_score={score:.3f}",
                source_url=source_url,
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
