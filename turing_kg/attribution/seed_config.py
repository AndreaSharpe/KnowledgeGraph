from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


NERBackend = Literal["spacy", "crf"]


@dataclass(frozen=True)
class SeedSpec:
    seed_id: str
    qid: str
    type: str = ""
    anchors_zh: tuple[str, ...] = ()
    anchors_en: tuple[str, ...] = ()
    triggers_zh: tuple[str, ...] = ()
    triggers_en: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttributionConfig:
    top_k: int = 1
    min_score: float = 0.25
    window_sentences: int = 0
    score_weights: dict[str, float] | None = None
    max_reasons_items: int = 8


@dataclass(frozen=True)
class ExtractionProfile:
    default_ner_backend: NERBackend = "spacy"
    crf_model_path: str = "models/crf_ner.pkl"
    per_source_overrides: dict[str, NERBackend] | None = None


@dataclass(frozen=True)
class EntityLinkingConfig:
    collective_enabled: bool = True
    collective_window_sentences: int = 2
    collective_top_k_candidates: int = 6
    collective_lambda_coherence: float = 0.35
    collective_coherence_props: tuple[str, ...] = ("P31", "P279", "P361", "P17", "P276", "P1416", "P166")
    collective_max_entities_to_fetch: int = 120


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_entities(project_root: Path) -> list[SeedSpec]:
    p = project_root / "sources" / "seed_entities.json"
    data = _read_json(p)
    out: list[SeedSpec] = []
    for s in data.get("seeds", []):
        out.append(
            SeedSpec(
                seed_id=str(s.get("seed_id", "")).strip(),
                qid=str(s.get("qid", "")).strip(),
                type=str(s.get("type", "")).strip(),
                anchors_zh=tuple(s.get("anchors_zh", []) or []),
                anchors_en=tuple(s.get("anchors_en", []) or []),
                triggers_zh=tuple(s.get("triggers_zh", []) or []),
                triggers_en=tuple(s.get("triggers_en", []) or []),
            )
        )
    return [x for x in out if x.seed_id]


def load_attribution_config(project_root: Path) -> AttributionConfig:
    p = project_root / "sources" / "attribution_config.json"
    if not p.is_file():
        return AttributionConfig()
    data = _read_json(p)
    return AttributionConfig(
        top_k=int(data.get("top_k", 1)),
        min_score=float(data.get("min_score", 0.25)),
        window_sentences=int(data.get("window_sentences", 0)),
        score_weights=dict(data.get("score_weights", {}) or {}),
        max_reasons_items=int(data.get("max_reasons_items", 8)),
    )


def load_extraction_profile(project_root: Path) -> ExtractionProfile:
    p = project_root / "sources" / "extraction_profile.json"
    if not p.is_file():
        return ExtractionProfile()
    data = _read_json(p)
    ov = data.get("per_source_overrides", {}) or {}
    # 这里不做严格校验，接入 build 时再兜底。
    return ExtractionProfile(
        default_ner_backend=str(data.get("default_ner_backend", "spacy")).strip() or "spacy",  # type: ignore[assignment]
        crf_model_path=str(data.get("crf_model_path", "models/crf_ner.pkl")).strip() or "models/crf_ner.pkl",
        per_source_overrides={str(k): str(v) for k, v in ov.items()} if isinstance(ov, dict) else {},
    )


def load_entity_linking_config(project_root: Path) -> EntityLinkingConfig:
    p = project_root / "sources" / "entity_linking_config.json"
    if not p.is_file():
        return EntityLinkingConfig()
    data = _read_json(p) or {}
    c = dict(data.get("collective", {}) or {})
    return EntityLinkingConfig(
        collective_enabled=bool(c.get("enabled", True)),
        collective_window_sentences=int(c.get("window_sentences", 2)),
        collective_top_k_candidates=int(c.get("top_k_candidates", 6)),
        collective_lambda_coherence=float(c.get("lambda_coherence", 0.35)),
        collective_coherence_props=tuple(c.get("coherence_props", []) or EntityLinkingConfig().collective_coherence_props),
        collective_max_entities_to_fetch=int(c.get("max_entities_to_fetch", 120)),
    )


def pick_ner_backend_for_source(profile: ExtractionProfile, source_key: str) -> NERBackend:
    if profile.per_source_overrides:
        # 1) 精确匹配 source_url
        if source_key in profile.per_source_overrides:
            v = profile.per_source_overrides[source_key]
            return "crf" if v == "crf" else "spacy"
        # 2) 兼容：配置用相对路径（raw/...#pages=1-50），但采集用 file:...#pages=1-50
        if source_key.startswith("file:"):
            # 只做最小兼容：把 file: 后面的绝对路径归一化成 raw/... 若包含项目的 raw 目录
            lowered = source_key.lower().replace("\\", "/")
            for k, v in profile.per_source_overrides.items():
                kk = str(k).lower().replace("\\", "/")
                if kk and kk in lowered:
                    return "crf" if v == "crf" else "spacy"
    return "crf" if profile.default_ner_backend == "crf" else "spacy"

