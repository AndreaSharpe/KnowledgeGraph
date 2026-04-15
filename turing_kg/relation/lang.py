"""中文占比与句子语言门控（与文档一致：默认阈值 0.3）。"""

from __future__ import annotations

import re


def zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / max(len(text), 1)


def is_zh_primary_sentence(text: str, *, min_ratio: float = 0.3) -> bool:
    return zh_ratio(text) >= min_ratio
