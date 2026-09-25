"""Comprueba si el clasificador aprendió atajos en lugar de "escritura de IA".

Dos pruebas:

1. Textos humanos de otra procedencia (data/smoke: resúmenes bajados de Crossref y de
   la API de arXiv). No pasaron por la reconstrucción del índice invertido de OpenAlex
   ni por la extracción de PDF, así que no tienen sus artefactos. Si el clasificador
   los marca como IA mucho más que a los humanos de test, está usando un atajo.

2. Normalización: se limpian los artefactos típicos de PDF (guiones de corte, espacios
   raros, comillas tipográficas) en los textos humanos de test. Si la puntuación cambia
   mucho, esos artefactos eran parte de lo que el modelo miraba.

Uso: python -m scripts.probe_shortcuts [--device cuda:1]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.common import CORPUS
from scripts.train_classifier import MAX_LEN, MODEL_DIR

SMOKE = Path(__file__).resolve().parent.parent / "data" / "smoke"


def normalize(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)          # guiones de corte de línea
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[“”]", '"', text).replace("’", "'")
    text = re.sub(r"\s+([,.;:])", r"\1", text)              # espacio antes de puntuación
    return re.sub(r"\s+", " ", text).strip()


@torch.inference_mode()
def score_texts(model, tok, texts, device, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        out += torch.softmax(logits, -1)[:, 1].tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--threshold-fpr", default="1.0%")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()

    test = [d for d in map(json.loads, (CORPUS / "dataset.jsonl").open()) if d["split"] == "test"]
    human_test = [d for d in test if d["label"] == "human"]
    p_human = score_texts(model, tok, [d["text"] for d in human_test], args.device)
    thr = float(np.quantile(p_human, 0.99))  # umbral 1% de falsos positivos EN test
    print(f"Umbral con 1% de falsos positivos sobre humanos de test: p_ai > {thr:.4f}\n")

    print("1) Humanos de otra procedencia (Crossref/arXiv, fase 1)")
    smoke_h = [json.loads(l) for l in (SMOKE / "human.jsonl").open()]
    p_smoke = score_texts(model, tok, [d["text"] for d in smoke_h], args.device)
    for lang in ("es", "en"):
        sel = p_smoke[[d["lang"] == lang for d in smoke_h]]
        print(f"   {lang}: n={len(sel)}  marcados como IA: {int((sel > thr).sum())} "
              f"({(sel > thr).mean():.0%})  p_ai mediana={np.median(sel):.4f}")
    smoke_ai = [json.loads(l) for l in (SMOKE / "ai.jsonl").open()]
    p_sai = score_texts(model, tok, [d["text"] for d in smoke_ai], args.device)
    print(f"   IA de la fase 1 detectada: {(p_sai > thr).mean():.0%} (n={len(p_sai)})")

    print("\n2) Humanos de test, con y sin normalización de artefactos")
    for kind in ("abstract", "thesis_passage"):
        sel = [d for d in human_test if d["kind"] == kind]
        base = score_texts(model, tok, [d["text"] for d in sel], args.device)
        norm = score_texts(model, tok, [normalize(d["text"]) for d in sel], args.device)
        print(f"   {kind:<15} n={len(sel):>5}  marcados: {(base > thr).mean():.1%} -> "
              f"{(norm > thr).mean():.1%}   |Δp| medio={np.abs(norm - base).mean():.4f}")


if __name__ == "__main__":
    main()
