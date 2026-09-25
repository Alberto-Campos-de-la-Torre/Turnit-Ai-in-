"""Evalúa Binoculars sobre una partición del dataset de la fase 2.

El umbral se fija sobre los textos humanos de la partición para una tasa de falsos
positivos dada, y se reporta qué fracción de IA se detecta, desglosado por idioma,
tipo de texto y generador. `ai_polished` se reporta aparte: no es "IA" ni "humano".

Uso: python -m scripts.eval_binoculars --split test
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
from scripts.common import CORPUS

FPRS = (0.01, 0.05)


def row(name, human, ai):
    y = np.r_[np.zeros(len(human)), np.ones(len(ai))]
    auroc = roc_auc_score(y, -np.r_[human, ai])  # más bajo = IA
    cells = [f"{name:<34} n_h={len(human):>5} n_ia={len(ai):>5}  AUROC={auroc:.3f}"]
    for fpr in FPRS:
        thr = np.quantile(human, fpr)
        cells.append(f"TPR@{fpr:.0%}={float((ai < thr).mean()):.2f}")
    print("  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--observer", default=DEFAULT_OBSERVER)
    ap.add_argument("--performer", default=DEFAULT_PERFORMER)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="qwen2.5-3b")
    args = ap.parse_args()

    out = CORPUS / f"scores_binoculars_{args.tag}_{args.split}.jsonl"
    docs = [d for d in map(json.loads, (CORPUS / "dataset.jsonl").open()) if d["split"] == args.split]
    cached = {}
    if out.exists():
        cached = {d["id"] + "|" + d["label"]: d["score"] for d in map(json.loads, out.open())}
    todo = [d for d in docs if d["id"] + "|" + d["label"] not in cached]
    if todo:
        bino = Binoculars(args.observer, args.performer, args.device, args.device)
        with out.open("a") as f:
            for i, d in enumerate(todo, 1):
                score = bino.score(d["text"])
                cached[d["id"] + "|" + d["label"]] = score
                f.write(json.dumps({"id": d["id"], "label": d["label"], "score": score}) + "\n")
                if i % 500 == 0:
                    print(f"  puntuados {i}/{len(todo)}", flush=True)
    for d in docs:
        d["score"] = cached[d["id"] + "|" + d["label"]]

    g = defaultdict(list)
    for d in docs:
        for key in [("all",), ("lang", d["lang"]), ("kind", d["kind"]), ("lang_kind", d["lang"], d["kind"]),
                    ("gen", d["generator"])]:
            g[(d["label"],) + key].append(d["score"])
    H = lambda *k: np.array(g[("human",) + k])

    print(f"\nBinoculars {args.observer} / {args.performer} — partición {args.split}\n")
    row("TODO (ai)", H("all"), np.array(g[("ai", "all")]))
    row("TODO (ai_polished)", H("all"), np.array(g[("ai_polished", "all")]))
    for lang in ("es", "en"):
        for kind in ("abstract", "thesis_passage"):
            row(f"{lang} / {kind}", H("lang_kind", lang, kind), np.array(g[("ai", "lang_kind", lang, kind)]))
    print()
    for gen in sorted({d["generator"] for d in docs if d["generator"]}):
        row(f"generador={gen}", H("all"), np.array(g[("ai", "gen", gen)]))
    print(f"\nPuntuaciones en {out}")


if __name__ == "__main__":
    main()
