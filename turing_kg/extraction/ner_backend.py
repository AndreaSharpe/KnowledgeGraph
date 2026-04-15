from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NERLabel = Literal["PER", "ORG", "LOC", "TIME", "DATE", "MONEY", "PERCENT"]


@dataclass(frozen=True)
class EntitySpan:
    mention: str
    label: NERLabel
    evidence_sentence: str
    start: int | None = None
    end: int | None = None


__all__ = ["EntitySpan", "NERLabel"]

