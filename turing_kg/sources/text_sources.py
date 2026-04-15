"""步骤二：采集非结构化/半结构化文本（维基摘要、粘贴文章、PDF 节选）。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .wikipedia_text import WikiChunk, fetch_seed_excerpts, fetch_turing_excerpts


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    text = raw.lstrip()
    if not text.startswith("---"):
        return {}, raw
    m = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$", text)
    if not m:
        return {}, raw
    fm, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta, body


_ARTICLE_BOUNDARY = re.compile(
    r"(?m)^(https://\S+|三联生活周刊.*?$|Alan Turing\(|纽约时报中文网$|BBC news中文$)",
)


def _infer_cite_url_title(chunk: str) -> tuple[str, str, str]:
    lines = [ln.strip() for ln in chunk.strip().splitlines() if ln.strip()]
    if not lines:
        return "user_excerpt", "", "excerpt"
    L0 = lines[0]
    if L0.startswith("http"):
        rest = L0.split(None, 1)
        url = rest[0]
        title_guess = rest[1][:120] if len(rest) > 1 else "article"
        return "zhihu_zhuanlan_turing_bio", url, title_guess
    if L0.startswith("三联生活周刊"):
        return "sanlian_lifeweek_turing_ai_father", "", L0[:120]
    if L0.startswith("Alan Turing"):
        return (
            "wikipedia_zh_snippet_alan_turing",
            "https://en.wikipedia.org/wiki/Alan_Turing",
            L0[:120],
        )
    if L0 == "纽约时报中文网" or L0.startswith("纽约时报"):
        for ln in lines[1:]:
            if len(ln) >= 10:
                return "nytcn_turing_morphogenesis_2018", "https://cn.nytimes.com/", ln[:120]
        return "nytcn_turing_morphogenesis_2018", "https://cn.nytimes.com/", "nyt"
    if "BBC" in L0 and "中文" in L0:
        for ln in lines[1:]:
            if len(ln) >= 10 and ("图灵" in ln or "英镑" in ln):
                return "bbc_cn_turing_banknote_2019", "https://www.bbc.com/zhongwen/simp", ln[:120]
        for ln in lines[1:]:
            if len(ln) >= 10:
                return "bbc_cn_turing_banknote_2019", "https://www.bbc.com/zhongwen/simp", ln[:120]
        return "bbc_cn_turing_banknote_2019", "https://www.bbc.com/zhongwen/simp", L0
    return "user_excerpt", "", L0[:120]


def split_book_excerpt_monolith(raw: str) -> list[tuple[str, str, str, str]]:
    text = raw.strip()
    if not text:
        return []
    meta, body = parse_front_matter(raw)
    if meta and body.strip() and "cite" in meta:
        return [
            (
                body.strip(),
                meta.get("cite", "user"),
                meta.get("url", ""),
                meta.get("title", "excerpt"),
            )
        ]

    matches = list(_ARTICLE_BOUNDARY.finditer(text))
    if not matches:
        return [(text, "user_excerpt_monolith", "", "book_excerpt")]

    chunks: list[tuple[str, str, str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        cite_key, url, title = _infer_cite_url_title(chunk)
        chunks.append((chunk, cite_key, url, title))
    return chunks


def chunks_from_pdf(project_root: Path) -> list[tuple[WikiChunk, str, str, str]]:
    cfg = project_root / "sources" / "pdf_sources.json"
    if not cfg.is_file():
        return []
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out: list[tuple[WikiChunk, str, str, str]] = []
    for spec in data.get("sources", []):
        if not spec.get("enabled"):
            continue
        rel = (spec.get("pdf_relative_path") or spec.get("path") or "").strip()
        if not rel:
            continue
        pdf_path = (project_root / rel).resolve()
        if not pdf_path.is_file():
            continue
        start = int(spec.get("page_start", 1))
        end = int(spec.get("page_end", start))
        from .pdf_text import extract_pdf_pages

        text = extract_pdf_pages(pdf_path, start, end)
        if not text.strip():
            continue
        cite = str(spec.get("cite_key", "hodges2012")).strip()
        label = str(spec.get("label", cite))[:200]
        src_url = f"file:{pdf_path.as_posix()}#pages={start}-{end}"
        out.append(
            (
                WikiChunk(lang="pdf", title=label, url=src_url, text=text),
                "pdf_excerpt",
                cite,
                src_url,
            )
        )
    return out


def chunks_from_article_dir(project_root: Path) -> list[tuple[WikiChunk, str, str, str]]:
    d = project_root / "raw" / "excerpts" / "articles"
    if not d.is_dir():
        return []
    out: list[tuple[WikiChunk, str, str, str]] = []
    for fp in sorted(d.glob("*.txt")):
        if fp.name.upper() == "README.TXT":
            continue
        raw = fp.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        if meta and body.strip():
            cite = meta.get("cite", fp.stem)
            url = meta.get("url", f"file:{fp.resolve().as_posix()}")
            title = meta.get("title", fp.stem)
            text_body = body.strip()
        else:
            cite, url, title = fp.stem, f"file:{fp.resolve().as_posix()}", fp.stem
            text_body = raw.strip()
        if not text_body:
            continue
        out.append(
            (
                WikiChunk(lang="article", title=title[:200], url=url, text=text_body),
                "article_file",
                cite,
                url,
            )
        )
    return out


def collect_text_sources(project_root: Path) -> list[tuple[WikiChunk, str, str, str]]:
    """返回 (WikiChunk, provenance标签, citation_key, source_url)。"""
    out: list[tuple[WikiChunk, str, str, str]] = []
    # Wikipedia 摘要：优先多 seed（若配置缺失则回退到仅图灵人物）
    seed_cfg = project_root / "sources" / "seed_entities.json"
    if seed_cfg.is_file():
        try:
            data = json.loads(seed_cfg.read_text(encoding="utf-8"))
            for ch in fetch_seed_excerpts(list(data.get("seeds", []) or [])):
                out.append((ch, "wikipedia", "", ch.url))
        except Exception:
            for ch in fetch_turing_excerpts():
                out.append((ch, "wikipedia", "", ch.url))
    else:
        for ch in fetch_turing_excerpts():
            out.append((ch, "wikipedia", "", ch.url))

    from_articles = chunks_from_article_dir(project_root)
    out.extend(from_articles)

    # 与 articles/*.txt 并行：只要存在 book_excerpt.txt 就纳入（不再二选一）
    be = project_root / "raw" / "excerpts" / "book_excerpt.txt"
    if be.is_file():
        raw = be.read_text(encoding="utf-8")
        p_url = f"file:{be.resolve().as_posix()}"
        for chunk_text, cite_key, url, title in split_book_excerpt_monolith(raw):
            final_url = url or p_url
            out.append(
                (
                    WikiChunk(lang="article", title=title[:200], url=final_url, text=chunk_text),
                    "book_excerpt",
                    cite_key.strip(),
                    final_url,
                )
            )

    out.extend(chunks_from_pdf(project_root))
    return out
