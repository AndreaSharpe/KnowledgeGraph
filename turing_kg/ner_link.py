"""步骤三：命名实体识别（spaCy）+ 实体链接（entity_linking）。"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ROOT_ENTITY_QID
from .entity_linking import link_mention_to_qid
from .graph_model import GraphBuild


@dataclass
class LinkedSpan:
    object_qid: str
    mention: str
    snippet: str
    score: float
    ner_label: str


def load_ner_link_config(project_root: Path) -> dict[str, Any]:
    p = project_root / "sources" / "ner_link_config.json"
    if not p.is_file():
        return {
            "root_anchors_zh": ["艾伦·图灵", "阿兰·图灵", "图灵"],
            "root_anchors_en": ["Alan Turing", "Alan Mathison Turing", "Turing"],
            "min_link_score": 0.14,
        }
    return json.loads(p.read_text(encoding="utf-8"))


_nlp_en = None
_nlp_zh = None


def _load_spacy():
    global _nlp_en, _nlp_zh
    try:
        import spacy
    except ImportError as e:
        raise ImportError("请安装：pip install spacy && python -m spacy download en_core_web_sm zh_core_web_sm") from e
    if _nlp_en is None:
        try:
            _nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            _nlp_en = None
            warnings.warn("缺少 en_core_web_sm", stacklevel=1)
    if _nlp_zh is None:
        try:
            _nlp_zh = spacy.load("zh_core_web_sm")
        except OSError:
            _nlp_zh = None
            warnings.warn("缺少 zh_core_web_sm", stacklevel=1)
    return _nlp_en, _nlp_zh


def _zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / max(len(text), 1)


def _split_sentences(text: str, use_zh: bool) -> list[str]:
    text = text.replace("\r", "\n")
    if use_zh:
        parts = re.split(r"(?<=[。！？；\n])", text)
    else:
        parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 8]


def _sentence_has_anchor(s: str, anchors: list[str]) -> bool:
    return any(a in s.strip() for a in anchors if a)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def extract_linked_spans(
    text: str,
    *,
    entity_map: dict[str, dict],
    root_anchors_zh: list[str],
    root_anchors_en: list[str],
    min_link_score: float,
    source_label: str,
) -> list[LinkedSpan]:
    use_zh = _zh_ratio(text) >= 0.12
    anchors = root_anchors_zh if use_zh else root_anchors_en
    nlp_en, nlp_zh = _load_spacy()
    nlp = nlp_zh if use_zh else nlp_en
    if nlp is None:
        nlp = nlp_en or nlp_zh
    if nlp is None:
        return []

    langs = ("zh", "en") if use_zh else ("en", "zh")
    sentences = _split_sentences(text, use_zh)
    out: list[LinkedSpan] = []
    seen_pair: set[tuple[str, str]] = set()
    allowed = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "NORP", "WORK_OF_ART", "PRODUCT"}

    for sent in sentences:
        if not _sentence_has_anchor(sent, anchors):
            continue
        doc = nlp(sent[:1_000_000])
        root_spans: list[tuple[int, int]] = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            et = ent.text.strip()
            if any(a in et or et in a for a in anchors if len(a) >= 2):
                root_spans.append((ent.start_char, ent.end_char))

        for ent in doc.ents:
            if ent.label_ not in allowed:
                continue
            span = (ent.start_char, ent.end_char)
            if any(_overlap(span, rs) for rs in root_spans) and ent.label_ == "PERSON":
                continue
            mention = ent.text.strip()
            if len(mention) < 2:
                continue
            qid, sc = link_mention_to_qid(
                mention,
                sent,
                languages=langs,
                min_score=min_link_score,
                entity_map_override=entity_map,
            )
            if not qid or qid == ROOT_ENTITY_QID:
                continue
            key = (qid, mention.lower())
            if key in seen_pair:
                continue
            seen_pair.add(key)
            out.append(
                LinkedSpan(
                    object_qid=qid,
                    mention=mention,
                    snippet=f"[{source_label}] {sent[:280]}",
                    score=sc,
                    ner_label=ent.label_,
                )
            )
    return out


def ingest_linked_spans(
    g: GraphBuild,
    spans: list[LinkedSpan],
    *,
    citation_key: str,
    source_url: str,
    root_qid: str = ROOT_ENTITY_QID,
) -> None:
    pred = "cooccurrence_linked"
    for sp in spans:
        g.ensure_node(root_qid)
        g.ensure_node(sp.object_qid, sp.mention[:120])
        g.add_edge(
            root_qid,
            sp.object_qid,
            f"EL_{sp.ner_label}",
            pred,
            "OUT",
            provenance="ner_entity_linking",
            citation_key=citation_key,
            snippet=f"{sp.snippet} | ner={sp.ner_label} score={sp.score:.2f}",
            source_url=source_url,
        )
