"""Calibra y valida el detector de mezcla de procedencias (detector/consistencia.py).

Mide la dispersión de la probabilidad por ventana en documentos de procedencia conocida:

- humanos puros y de IA puros: deberían dar dispersión baja (una sola mano),
- mixtos al 30% y al 60%: deberían darla alta,
- y los documentos reales del usuario, que son el caso a observar.

Imprime el AUROC, el umbral que deja un 5% de falsos positivos en documentos puros y el
resultado de aplicarlo.

Uso: python -m scripts.eval_consistencia [--n 20] [--device cuda:1]
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.consistencia import MIN_VENTANAS, analizar
from detector.pipeline import MODEL_DIR
from scripts.common import CORPUS
from scripts.eval_documents import build_docs
from scripts.probe_shortcuts import score_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    from detector.binoculars import split_sentences

    def ventanas(texto, palabras=150):
        oraciones = [texto[a:b] for a, b in split_sentences(texto)]
        salida, actual, cuenta = [], [], 0
        for o in oraciones:
            actual.append(o)
            cuenta += len(o.split())
            if cuenta >= palabras:
                salida.append(" ".join(actual))
                actual, cuenta = [], 0
        if actual:
            salida.append(" ".join(actual))
        return salida

    def consistencia(texto):
        v = ventanas(texto)
        if len(v) < MIN_VENTANAS:
            return None
        return analizar(score_texts(clf, tok, v, args.device))

    grupos = defaultdict(list)
    for d in build_docs(args.n, random.Random(11)):
        c = consistencia(d["text"])
        if c and not c.aviso:
            grupos[d["tipo"]].append(c)

    puros = [c.dispersion for c in grupos["humano"] + grupos["ia"]]
    mixtos = [c.dispersion for c in grupos["mixto-30"] + grupos["mixto-60"]]
    y = np.r_[np.zeros(len(puros)), np.ones(len(mixtos))]
    print(f"\nAUROC de la dispersión (mixto vs puro) = {roc_auc_score(y, np.r_[puros, mixtos]):.3f}")
    print("criterio aplicado: racha de 2+ fragmentos marcados y 2+ claramente humanos\n")
    print(f"{'tipo de documento':<22}{'n':>4}{'marcados como mezcla':>23}")
    for tipo in ("humano", "ia", "mixto-30", "mixto-60"):
        v = grupos[tipo]
        if v:
            print(f"{tipo:<22}{len(v):>4}{np.mean([c.mezcla for c in v]):>22.0%}")

    reales = CORPUS / "reales_ia.jsonl"
    if reales.exists():
        print("\ndocumentos reales del usuario (escritos con IA en conversación):")
        for d in map(json.loads, reales.open()):
            c = consistencia(d["text"])
            if c is None or c.aviso:
                print(f"  {d['archivo'][:44]:<46} (muy corto)")
            else:
                print(f"  {d['archivo'][:44]:<46} {c.marcados}/{c.ventanas} de máquina, "
                      f"racha {c.racha} → {c.etiqueta}")


if __name__ == "__main__":
    main()
