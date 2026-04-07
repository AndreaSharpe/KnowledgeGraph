"""
知识图谱构建（教科书式流水线）

1. 结构化知识：Wikidata API（中心实体陈述）
2. 文本采集：维基摘要、book_excerpt / articles、PDF 节选
3. 信息抽取：spaCy NER → 实体链接（Wikidata 搜索 + 重排序）→ 写入图
4. 导出：nodes.csv、relationships.csv、graph_summary.json、triples_extracted.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .config import ROOT_ENTITY_QID
from .export_io import write_graph_csv_json, write_triples_csv
from .graph_model import GraphBuild
from .ner_link import (
    extract_linked_spans,
    ingest_linked_spans,
    load_ner_link_config,
)
from .sources_io import load_bibliography, load_entity_map
from .wikidata_layer import load_structured_graph


def build_knowledge_graph(project_root: Path) -> tuple[GraphBuild, list[dict[str, str]]]:
    """执行步骤 1–3，返回图与用于 triples CSV 的行列表。"""
    # --- 步骤 1：Wikidata 结构化层 ---
    g = load_structured_graph()

    emap = load_entity_map(project_root / "sources" / "entity_map.csv")
    ncfg = load_ner_link_config(project_root)
    anchors_zh = list(ncfg.get("root_anchors_zh", []))
    anchors_en = list(ncfg.get("root_anchors_en", []))
    min_ls = float(ncfg.get("min_link_score", 0.14))

    # --- 步骤 2–3：文本 + NER + 实体链接 ---
    from .text_sources import collect_text_sources

    triple_rows: list[dict[str, str]] = []

    for ch, _prov, cite_key, url in collect_text_sources(project_root):
        label = f"{ch.lang}:{ch.title}"
        try:
            spans = extract_linked_spans(
                ch.text,
                entity_map=emap,
                root_anchors_zh=anchors_zh,
                root_anchors_en=anchors_en,
                min_link_score=min_ls,
                source_label=label,
            )
        except ImportError as e:
            warnings.warn(str(e), stacklevel=1)
            spans = []

        ingest_linked_spans(g, spans, citation_key=cite_key, source_url=url)

        for sp in spans:
            triple_rows.append(
                {
                    "subject_qid": ROOT_ENTITY_QID,
                    "predicate": "cooccurrence_linked",
                    "object_mention": sp.mention,
                    "object_qid": sp.object_qid,
                    "ner_label": sp.ner_label,
                    "link_score": f"{sp.score:.4f}",
                    "citation_key": cite_key,
                    "source_url": url,
                    "snippet": sp.snippet,
                }
            )

    return g, triple_rows


def export_all(
    project_root: Path,
    g: GraphBuild,
    triple_rows: list[dict[str, str]],
) -> None:
    """步骤 4：导出到 data/。"""
    data_dir = project_root / "data"
    bib = load_bibliography(project_root / "sources" / "bibliography.json")
    write_graph_csv_json(g, data_dir, bibliography=bib)
    write_triples_csv(data_dir / "triples_extracted.csv", triple_rows)
