#!/usr/bin/env python3
"""
抓取高质量网页语料到 raw/excerpts/articles/*.txt（用于提升 DS/RE 的有效样本量）。

输入：
- sources/web_sources.json

输出：
- raw/excerpts/articles/web_{id}.txt（带 front-matter：cite/url/title）

实现原则：
- 离线可复现：所有抓取 URL 都来自 web_sources.json
- 不依赖重型解析库（仅 requests + 简单清洗）；遇到反爬/动态站点可改为手工粘贴
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE).strip("_")
    return s[:80] or "source"


def _strip_html(html: str) -> str:
    # 极简 HTML 去标签：适配 Wikipedia/普通静态页面；对复杂站点效果一般，但足够做语料补充。
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\\1>", " ", html)
    html = re.sub(r"(?is)<br\\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p\\s*>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"[ \\t\\f\\v]+", " ", html)
    text = re.sub(r"\\n\\s*\\n\\s*\\n+", "\n\n", text)
    return text.strip()


def _fetch(url: str, *, timeout: tuple[float, float] = (30.0, 120.0)) -> str:
    r = requests.get(url, headers={"User-Agent": "KnowledgeGraphCourseBot/1.0"}, timeout=timeout)
    r.raise_for_status()
    # requests 会根据响应 header 推断 encoding
    return r.text


def _as_article_txt(*, cite: str, url: str, title: str, body: str) -> str:
    fm = "\n".join(
        [
            "---",
            f"cite: {cite}",
            f"url: {url}",
            f"title: {title}",
            "---",
            "",
        ]
    )
    return fm + body.strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="抓取 web_sources.json 中的网页到 raw/excerpts/articles/")
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--only-id", type=str, default="", help="只抓取指定 id（web_sources.json 里的 id）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    root = args.project_root.resolve()
    cfg_path = root / "sources" / "web_sources.json"
    if not cfg_path.is_file():
        raise SystemExit(f"未找到配置：{cfg_path}")
    cfg = _read_json(cfg_path) or {}
    items = list(cfg.get("sources") or [])
    if args.only_id:
        items = [x for x in items if str(x.get("id") or "") == args.only_id]
    items = [x for x in items if bool(x.get("enabled", True)) and str(x.get("url") or "").startswith("http")]
    if not items:
        raise SystemExit("没有可抓取的 sources。")

    out_dir = root / "raw" / "excerpts" / "articles"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for it in items:
        sid = str(it.get("id") or "source")
        url = str(it.get("url") or "")
        title = str(it.get("title") or sid)
        cite = str(it.get("cite_key") or sid)
        out_path = out_dir / f"web_{_slug(sid)}.txt"
        if out_path.is_file() and not args.force:
            print(f"skip exists: {out_path.name}")
            continue
        try:
            html = _fetch(url)
            text = _strip_html(html)
            # 轻量截断：避免极端超长页面
            text = text[:250_000]
            out_path.write_text(_as_article_txt(cite=cite, url=url, title=title, body=text), encoding="utf-8")
            ok += 1
            print(f"ok: {sid} -> {out_path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"fail: {sid} url={url} err={e}", file=sys.stderr)

    print(f"done: {ok}/{len(items)}")


if __name__ == "__main__":
    main()

