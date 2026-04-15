from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def split_zh_sentences(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 8]


def to_conll_chars(sent: str) -> str:
    lines = []
    for ch in sent.strip():
        if ch == "\n":
            continue
        lines.append(f"{ch}\tO")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="raw/艾伦·图灵传：如谜的解谜者.pdf")
    ap.add_argument("--page-start", type=int, default=1)
    ap.add_argument("--page-end", type=int, default=50)
    ap.add_argument("--out-dir", default="data/ner_crf")
    ap.add_argument("--max-sents", type=int, default=200)
    args = ap.parse_args()

    project_root = Path(".").resolve()
    pdf_path = (project_root / args.pdf).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    from turing_kg.sources.pdf_text import extract_pdf_pages

    text = extract_pdf_pages(pdf_path, args.page_start, args.page_end)
    sents = split_zh_sentences(text)
    sents = sents[: max(0, int(args.max_sents))]

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写出一个可人工修改的 train/dev 初稿（先按 8:2 切）
    n = len(sents)
    cut = int(n * 0.8)
    train_s = sents[:cut]
    dev_s = sents[cut:]

    def dump(path: Path, xs: list[str]) -> None:
        parts = []
        for s in xs:
            parts.append(to_conll_chars(s))
            parts.append("")  # sentence boundary
        path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")

    dump(out_dir / "train.conll", train_s)
    dump(out_dir / "dev.conll", dev_s)

    (out_dir / "sample_sentences.json").write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "pages": [args.page_start, args.page_end],
                "sentence_count": n,
                "max_sents": args.max_sents,
                "note": "train/dev 是 O 标注模板，请按 BIO 七类人工改标后训练。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote -> {out_dir / 'train.conll'}")
    print(f"wrote -> {out_dir / 'dev.conll'}")
    print(f"meta  -> {out_dir / 'sample_sentences.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

