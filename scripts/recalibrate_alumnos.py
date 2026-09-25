"""Recalibra el umbral usando escritura de ALUMNOS como referencia humana.

Hasta ahora el umbral salía de los textos humanos de validación, que son artículos y
tesis publicados: prosa pulida y revisada. Lo que el profesor analiza son trabajos de
alumnos, así que el umbral se fija ahora sobre tres corpus de escritura estudiantil
real y anterior a 2022:

- CATyPI: secciones de tesis de computación en español (INAOE).
- Tesis en español descargadas aparte, que no están en el corpus de entrenamiento.
- PERSUADE 2.0: ensayos escolares en inglés.

Los pesos del ensamble no se tocan (se ajustaron en validación); solo se eligen de nuevo
los umbrales, que son el punto de operación.

Uso: python -m scripts.recalibrate_alumnos [--device cuda:0] [--apply]
"""
import argparse
import json
import math

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
from scripts.calibrate import CAL_PATH
from scripts.common import CORPUS
from scripts.probe_shortcuts import score_texts
from scripts.train_classifier import MODEL_DIR

FUENTES = [("catypi", "alumnos_catypi.jsonl"),
           ("tesis_es", "alumnos_tesis_es.jsonl"),
           ("persuade_en", "alumnos_persuade.jsonl")]
MIN_PALABRAS = 150
FPRS = (0.005, 0.01, 0.05)


def cargar():
    filas = []
    for nombre, archivo in FUENTES:
        ruta = CORPUS / archivo
        if not ruta.exists():
            print(f"  (falta {archivo})")
            continue
        for d in map(json.loads, ruta.open()):
            if len(d["text"].split()) >= MIN_PALABRAS:
                filas.append({"corpus": nombre, "text": d["text"]})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--apply", action="store_true", help="escribe los umbrales nuevos en calibration.json")
    args = ap.parse_args()

    filas = cargar()
    print(f"{len(filas)} textos de alumnos con {MIN_PALABRAS}+ palabras")
    cal = json.loads(CAL_PATH.read_text())
    (w_clf, w_bino), b = cal["coef"], cal["intercept"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    p = np.clip(score_texts(clf, tok, [f["text"] for f in filas], args.device), 1e-6, 1 - 1e-6)
    del clf
    torch.cuda.empty_cache()

    bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, args.device, args.device)
    b_scores = np.array([bino.score(f["text"]) for f in filas])
    scores = w_clf * np.log(p / (1 - p)) + w_bino * b_scores + b

    corpus = np.array([f["corpus"] for f in filas])
    json.dump({"corpus": corpus.tolist(), "scores": scores.tolist()},
              (CORPUS / "scores_alumnos.json").open("w"))
    nuevos = {f"{f:.1%}": float(np.quantile(scores, 1 - f)) for f in FPRS}

    print(f"\n{'corpus':<14}{'n':>6}{'umbral 1%':>12}{'umbral 0,5%':>13}   (el que daría cada corpus por su cuenta)")
    for nombre, _ in FUENTES:
        m = corpus == nombre
        if m.sum():
            print(f"{nombre:<14}{m.sum():>6}{np.quantile(scores[m], 0.99):>12.2f}{np.quantile(scores[m], 0.995):>13.2f}")
    print("\numbrales actuales (calibrados con artículos):",
          {k: round(v, 2) for k, v in cal["thresholds"].items()})
    print("umbrales nuevos  (calibrados con alumnos):   ",
          {k: round(v, 2) for k, v in nuevos.items()})

    print(f"\n{'corpus':<14}{'n':>6}   por encima del umbral, con cada calibración")
    for nombre, _ in FUENTES:
        m = corpus == nombre
        if not m.sum():
            continue
        viejo = float((scores[m] > cal["thresholds"]["1.0%"]).mean())
        nuevo = float((scores[m] > nuevos["1.0%"]).mean())
        print(f"{nombre:<14}{m.sum():>6}   antes={viejo:>6.2%}   ahora={nuevo:>6.2%}")

    if args.apply:
        cal["thresholds_articulos"] = cal["thresholds"]
        cal["thresholds"] = nuevos
        cal["calibrado_con"] = f"escritura estudiantil: {[n for n, _ in FUENTES]} (n={len(filas)})"
        CAL_PATH.write_text(json.dumps(cal, indent=2))
        print(f"\nCalibración actualizada -> {CAL_PATH}")
    else:
        print("\n(no se escribió nada; usa --apply para guardar)")


if __name__ == "__main__":
    main()
