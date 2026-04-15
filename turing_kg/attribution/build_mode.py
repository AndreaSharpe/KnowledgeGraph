"""构建模式：full / export_only / from_curated。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

BuildMode = Literal["full", "export_only", "from_curated"]


def load_build_mode(project_root: Path) -> BuildMode:
    p = project_root / "sources" / "build_config.json"
    if not p.is_file():
        return "full"
    data = json.loads(p.read_text(encoding="utf-8"))
    m = str(data.get("mode", "full")).strip().lower()
    if m in ("full", "export_only", "from_curated"):
        return m  # type: ignore[return-value]
    return "full"
