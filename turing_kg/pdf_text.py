from __future__ import annotations

import warnings
from pathlib import Path


def extract_pdf_pages(path: Path, page_start: int, page_end: int) -> str:
    """从 PDF 抽取可选中复制的文字层（页码从 1 起，含首尾页）。

    若全书为扫描图片、未做 OCR，则返回空字符串——请先对 PDF 做 OCR，
    或把获准使用的章节打字/复制到 raw/excerpts/book_excerpt.txt。
    """
    path = Path(path)
    a = max(1, int(page_start))
    b = int(page_end)

    def via_pypdf() -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        n = len(reader.pages)
        if n == 0:
            return ""
        hi = min(b, n)
        if a > hi:
            return ""
        parts: list[str] = []
        for i in range(a - 1, hi):
            t = reader.pages[i].extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    def via_pymupdf() -> str:
        import fitz

        doc = fitz.open(str(path))
        n = doc.page_count
        if n == 0:
            return ""
        hi = min(b, n)
        if a > hi:
            return ""
        parts: list[str] = []
        for i in range(a - 1, hi):
            t = doc[i].get_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    text = via_pypdf().strip()
    if text:
        return text
    try:
        text = via_pymupdf().strip()
    except ImportError:
        text = ""
    if not text:
        warnings.warn(
            f"PDF 在指定页码未抽到文字（常见原因：扫描版无文字层）。文件：{path}",
            stacklevel=2,
        )
    return text
