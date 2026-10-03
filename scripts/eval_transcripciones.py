"""Mide el detector sobre la prosa real escrita con Claude en conversación.

Es el caso extremo: el autor dirige turno a turno, así que el contenido y las decisiones
son humanas y solo el fraseo es de máquina. Línea base (modelo de revisión):
p_ia mediana 0.0003, 18% de los bloques detectados, y Binoculars los puntúa MÁS humanos
que el texto humano real.

Uso: python -m scripts.eval_transcripciones [--device cuda:1]
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
from scripts.calibrate import CAL_PATH
from scripts.common import CORPUS
from scripts.probe_shortcuts import score_texts
from scripts.train_classifier import MODEL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    filas = [json.loads(l) for l in (CORPUS / "reales_transcripciones.jsonl").open()]
    textos = [d["text"] for d in filas]
    cal = json.loads(CAL_PATH.read_text())
    (w_clf, w_bino), b = cal["coef"], cal["intercept"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    p = np.clip(score_texts(clf, tok, textos, args.device), 1e-6, 1 - 1e-6)
    del clf
    torch.cuda.empty_cache()

    bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, args.device, args.device)
    bb = np.array([bino.score(t) for t in textos])
    ensamble = w_clf * np.log(p / (1 - p)) + w_bino * bb + b

    print(f"\nprosa real escrita con Claude en conversación (n={len(filas)})\n")
    print(f"  p_clasificador mediana: {np.median(p):.4f}   detectados (p>0.5): {(p > 0.5).mean():.0%}")
    print(f"  binoculars mediana:     {np.median(bb):.3f}")
    for nivel, umbral in cal["thresholds"].items():
        print(f"  ensamble, umbral {nivel:<5}: marcados {(ensamble > umbral).mean():.0%}")


if __name__ == "__main__":
    main()
