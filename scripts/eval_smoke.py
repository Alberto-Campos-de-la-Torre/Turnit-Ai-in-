"""Evalúa Binoculars sobre data/smoke/{human,ai}.jsonl.

Reporta AUROC y, para varios umbrales fijados sobre los textos humanos, qué fracción
de texto IA se detecta. Con ~50 textos humanos, una tasa de falsos positivos del 1%
no se puede estimar: estas cifras solo sirven para validar que el método funciona.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars

DATA = Path(__file__).resolve().parent.parent / "data" / "smoke"


def report(name: str, human: np.ndarray, ai: np.ndarray):
    # Binoculars: más bajo = IA, así que se invierte el signo para el AUROC.
    y = np.r_[np.zeros(len(human)), np.ones(len(ai))]
    auroc = roc_auc_score(y, -np.r_[human, ai])
    line = f"{name:<28} n_h={len(human):>3} n_ia={len(ai):>3}  AUROC={auroc:.3f}"
    for fpr in (0.0, 0.05, 0.10):
        thr = np.quantile(human, fpr) if fpr else human.min()
        tpr = float((ai < thr).mean())
        line += f"  TPR@FPR{int(fpr * 100)}%={tpr:.2f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default=DEFAULT_OBSERVER)
    ap.add_argument("--performer", default=DEFAULT_PERFORMER)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="qwen2.5-3b")
    args = ap.parse_args()

    docs = [json.loads(l) for f in ("human.jsonl", "ai.jsonl") for l in (DATA / f).open()]
    bino = Binoculars(args.observer, args.performer, args.device, args.device)
    for d in docs:
        d["score"] = bino.score(d["text"])

    out = DATA / f"scores_{args.tag}.jsonl"
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps({k: d[k] for k in ("id", "lang", "label", "generator", "score")}) + "\n")

    groups = defaultdict(list)
    for d in docs:
        groups[(d["label"], d["lang"], d["generator"])].append(d["score"])

    def pick(label, lang=None, gen=None):
        return np.array([s for (l, g_lang, g), v in groups.items() for s in v
                         if l == label and lang in (None, g_lang) and gen in (None, g)])

    print(f"\nModelos: {args.observer} / {args.performer}")
    for lang in ("es", "en"):
        h = pick("human", lang)
        print(f"  humano {lang}: media={h.mean():.3f} min={h.min():.3f} max={h.max():.3f}")
        for gen in sorted({g for (l, _, g) in groups if l == "ai"}):
            a = pick("ai", lang, gen)
            print(f"  {gen:<12} {lang}: media={a.mean():.3f} min={a.min():.3f} max={a.max():.3f}")
    print()
    report("TODO", pick("human"), pick("ai"))
    for lang in ("es", "en"):
        report(f"idioma={lang}", pick("human", lang), pick("ai", lang))
    for gen in sorted({g for (l, _, g) in groups if l == "ai"}):
        report(f"generador={gen}", pick("human"), pick("ai", gen=gen))
    print(f"\nPuntuaciones guardadas en {out}")


if __name__ == "__main__":
    main()
