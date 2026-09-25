"""Combina clasificador + Binoculars y calibra el umbral.

La regresión logística se ajusta en validación y se evalúa en test; los umbrales se
eligen sobre los textos HUMANOS de validación para una tasa de falsos positivos dada.
Aplicarlos a test da una estimación honesta: ningún umbral se elige mirando test.

Salida: models/calibration.json con los pesos, los umbrales y las métricas.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.common import CORPUS
from scripts.train_classifier import MODEL_DIR

CAL_PATH = MODEL_DIR.parent / "calibration.json"
FPRS = (0.005, 0.01, 0.05)


def load(split, tag="qwen2.5-3b"):
    """Devuelve las filas del dataset con las dos puntuaciones ya unidas."""
    bino = {f"{d['id']}|{d['label']}": d["score"]
            for d in map(json.loads, (CORPUS / f"scores_binoculars_{tag}_{split}.jsonl").open())}
    clf = {f"{d['id']}|{d['label']}": d["p_ai"]
           for d in map(json.loads, (CORPUS / f"scores_clf_{split}.jsonl").open())}
    rows = []
    for d in map(json.loads, (CORPUS / "dataset.jsonl").open()):
        if d["split"] != split:
            continue
        k = f"{d['id']}|{d['label']}"
        if k in bino and k in clf:
            d["bino"], d["p_clf"] = bino[k], clf[k]
            rows.append(d)
    return rows


def features(rows):
    p = np.clip([r["p_clf"] for r in rows], 1e-6, 1 - 1e-6)
    return np.c_[np.log(p / (1 - p)), [r["bino"] for r in rows]]


def metrics(name, human, ai, thresholds):
    if len(ai) == 0 or len(human) == 0:
        return
    auroc = roc_auc_score(np.r_[np.zeros(len(human)), np.ones(len(ai))], np.r_[human, ai])
    cells = [f"{name:<34} n_h={len(human):>5} n_ia={len(ai):>5}  AUROC={auroc:.4f}"]
    for fpr, thr in thresholds.items():
        cells.append(f"TPR@{fpr}={float((ai > thr).mean()):.3f}")
    print("  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen2.5-3b")
    args = ap.parse_args()

    val, test = load("val", args.tag), load("test", args.tag)
    fit = [r for r in val if r["label"] in ("human", "ai")]
    y = np.array([r["label"] == "ai" for r in fit], dtype=int)
    model = LogisticRegression(max_iter=1000).fit(features(fit), y)

    def score(rows):
        return model.decision_function(features(rows))

    val_scores, test_scores = score(val), score(test)
    val_human = val_scores[[r["label"] == "human" for r in val]]
    thresholds = {f"{f:.1%}": float(np.quantile(val_human, 1 - f)) for f in FPRS}

    print("Pesos: clasificador=%.3f  binoculars=%.3f  sesgo=%.3f"
          % (model.coef_[0][0], model.coef_[0][1], model.intercept_[0]))
    print("Umbrales (elegidos en validación):",
          {k: round(v, 3) for k, v in thresholds.items()}, "\n")

    for split, rows, scores in (("val", val, val_scores), ("test", test, test_scores)):
        by = defaultdict(list)
        for r, s in zip(rows, scores):
            by[(r["label"], "all")].append(s)
            by[(r["label"], r["lang"], r["kind"])].append(s)
            by[(r["label"], "gen", r["generator"])].append(s)
        h = np.array(by[("human", "all")])
        print(f"— partición {split} —")
        metrics("ensamble (ai)", h, np.array(by[("ai", "all")]), thresholds)
        metrics("ensamble (ai_polished)", h, np.array(by[("ai_polished", "all")]), thresholds)
        if split == "test":
            for lang in ("es", "en"):
                for kind in ("abstract", "thesis_passage"):
                    metrics(f"  {lang} / {kind}", np.array(by[("human", lang, kind)]),
                            np.array(by[("ai", lang, kind)]), thresholds)
            for gen in sorted({r["generator"] for r in rows if r["generator"]}):
                metrics(f"  generador={gen}", h, np.array(by[("ai", "gen", gen)]), thresholds)
        print()

    CAL_PATH.write_text(json.dumps({
        "features": ["logit_p_clf", "binoculars"],
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "thresholds": thresholds, "binoculars_tag": args.tag,
    }, indent=2))
    print(f"Calibración -> {CAL_PATH}")


if __name__ == "__main__":
    main()
