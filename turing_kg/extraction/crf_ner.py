from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ner_backend import EntitySpan, NERLabel


_LABEL_MAP: dict[str, NERLabel] = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "TIME": "TIME",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
}


def zh_char_tokenize(sentence: str) -> list[str]:
    """
    中文默认字级 tokenization（经典 CRF 做法，降低分词依赖）。
    这里做极少量清洗：压缩空白，保留标点作为 token。
    """
    s = re.sub(r"\s+", " ", sentence).strip()
    if not s:
        return []
    return list(s)


def _shape(ch: str) -> str:
    if re.fullmatch(r"[0-9]", ch):
        return "D"
    if re.fullmatch(r"[A-Za-z]", ch):
        return "A"
    if re.fullmatch(r"[\u4e00-\u9fff]", ch):
        return "H"
    if re.fullmatch(r"\s", ch):
        return "S"
    return "P"


def featurize_zh_chars(tokens: list[str], i: int) -> dict[str, Any]:
    """
    经典 CRF 手工特征：当前字/前后字/字形等。
    """
    tok = tokens[i]
    feats: dict[str, Any] = {
        "bias": 1.0,
        "ch": tok,
        "shape": _shape(tok),
        "is_digit": bool(re.fullmatch(r"[0-9]", tok)),
        "is_hanzi": bool(re.fullmatch(r"[\u4e00-\u9fff]", tok)),
    }
    if i > 0:
        p = tokens[i - 1]
        feats.update({"-1:ch": p, "-1:shape": _shape(p)})
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        n = tokens[i + 1]
        feats.update({"+1:ch": n, "+1:shape": _shape(n)})
    else:
        feats["EOS"] = True
    # 简单二元组
    if 0 < i < len(tokens) - 1:
        feats["-1:ch+ch"] = tokens[i - 1] + tok
        feats["ch+1:ch"] = tok + tokens[i + 1]
    return feats


@dataclass(frozen=True)
class CRFModel:
    """训练脚本输出的模型容器的最小约定。"""

    tagger: Any
    labels: tuple[str, ...]


def load_crf_model(model_path: str | Path) -> CRFModel:
    p = Path(model_path)
    obj = pickle.loads(p.read_bytes())
    # 约定：训练脚本保存的是 CRFModel 或兼容 dict
    if isinstance(obj, CRFModel):
        return obj
    if isinstance(obj, dict) and "tagger" in obj:
        labels = tuple(obj.get("labels", []) or [])
        return CRFModel(tagger=obj["tagger"], labels=labels)
    raise TypeError("无法识别 CRF 模型格式；请用 scripts/train_crf_ner.py 训练生成。")


def bio_to_entities(tokens: list[str], tags: list[str], sentence: str) -> list[EntitySpan]:
    """
    BIO tags -> entity spans（以 token 索引为准；字符级时可近似映射为字符串片段）。
    这里返回的 start/end 为 token 索引（后续接入时可扩展为字符 offset）。
    """
    out: list[EntitySpan] = []
    i = 0
    while i < len(tokens) and i < len(tags):
        t = tags[i]
        if not t or t == "O":
            i += 1
            continue
        if "-" not in t:
            i += 1
            continue
        pref, lab = t.split("-", 1)
        if pref != "B":
            i += 1
            continue
        lab2 = _LABEL_MAP.get(lab)
        if not lab2:
            i += 1
            continue
        j = i + 1
        while j < len(tokens) and j < len(tags):
            tj = tags[j]
            if tj == f"I-{lab}":
                j += 1
                continue
            break
        mention = "".join(tokens[i:j]).strip()
        if mention:
            out.append(
                EntitySpan(
                    mention=mention,
                    label=lab2,
                    evidence_sentence=sentence,
                    start=i,
                    end=j,
                )
            )
        i = j
    return out


def predict_sentence_entities_zh(model: CRFModel, sentence: str) -> list[EntitySpan]:
    toks = zh_char_tokenize(sentence)
    if not toks:
        return []
    feats = [featurize_zh_chars(toks, i) for i in range(len(toks))]
    # 约定：tagger 具备 tag(xseq) -> yseq（sklearn-crfsuite 兼容）
    tags = list(model.tagger.tag(feats))
    return bio_to_entities(toks, tags, sentence)


def extract_entities_with_crf_zh(
    text_sentences: Iterable[str],
    *,
    model_path: str | Path,
) -> list[EntitySpan]:
    model = load_crf_model(model_path)
    out: list[EntitySpan] = []
    for s in text_sentences:
        out.extend(predict_sentence_entities_zh(model, s))
    return out

