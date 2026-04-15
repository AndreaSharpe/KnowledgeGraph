from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

import sklearn_crfsuite
from seqeval.metrics import f1_score

from turing_kg.extraction.crf_ner import CRFModel, featurize_zh_chars, zh_char_tokenize


def read_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    """
    读取 CoNLL: TOKEN<TAB>LABEL，句间空行。
    返回 (tokens, labels) 序列列表。
    """
    sents: list[list[str]] = []
    labs: list[list[str]] = []
    cur_t: list[str] = []
    cur_l: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln:
            if cur_t:
                sents.append(cur_t)
                labs.append(cur_l)
            cur_t, cur_l = [], []
            continue
        if "\t" not in ln:
            continue
        tok, lab = ln.split("\t", 1)
        tok = tok.strip()
        lab = lab.strip()
        if not tok:
            continue
        cur_t.append(tok)
        cur_l.append(lab)
    if cur_t:
        sents.append(cur_t)
        labs.append(cur_l)
    return list(zip(sents, labs))


def to_features(tokens: list[str]) -> list[dict]:
    return [featurize_zh_chars(tokens, i) for i in range(len(tokens))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/ner_crf/train.conll")
    ap.add_argument("--dev", default="data/ner_crf/dev.conll")
    ap.add_argument("--out", default="models/crf_ner.pkl")
    args = ap.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = read_conll(train_path)
    dev = read_conll(dev_path)

    X_train = [to_features(toks) for toks, _ in train]
    y_train = [labs for _, labs in train]

    X_dev = [to_features(toks) for toks, _ in dev]
    y_dev = [labs for _, labs in dev]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_dev)
    f1 = f1_score(y_dev, y_pred)
    print(f"dev entity_f1(seqeval)={f1:.4f}")

    labels = tuple(sorted({lab for seq in y_train for lab in seq if lab != 'O'}))
    model = CRFModel(tagger=crf, labels=labels)
    out_path.write_bytes(pickle.dumps(model))
    print(f"saved model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

