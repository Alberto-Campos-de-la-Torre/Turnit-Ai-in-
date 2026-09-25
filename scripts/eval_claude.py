"""Evalúa el detector sobre los textos escritos por Claude (generador no visto).

Lee data/corpus/claude_texts.jsonl ({"job_id", "texto"}), lo valida, lo puntúa con el
clasificador y con Binoculars, aplica el umbral ya calibrado en validación y reporta
cuánto detecta, por idioma, tipo y tarea.

Uso: python -m scripts.eval_claude [--device cuda:1]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
from scripts.calibrate import CAL_PATH
from scripts.common import CORPUS, detect_lang
from scripts.finalize_corpus import REFUSAL, clean_ai
from scripts.probe_shortcuts import score_texts
from scripts.train_classifier import MODEL_DIR


def load_jobs_and_texts(jobs_file="claude_jobs.jsonl", texts_file="claude_texts.jsonl"):
    jobs = {j["job_id"]: j for j in map(json.loads, (CORPUS / jobs_file).open())}
    rows, drops = [], defaultdict(int)
    for d in map(json.loads, (CORPUS / texts_file).open()):
        j = jobs.get(d["job_id"])
        text = clean_ai(d.get("texto", ""))
        if j is None:
            drops["job_id desconocido"] += 1
        elif not text or REFUSAL.search(text):
            drops["vacío o negativa"] += 1
        elif detect_lang(text) != j["idioma"]:
            drops["idioma incorrecto"] += 1
        elif not 0.5 * j["palabras"] <= len(text.split()) <= 2.0 * j["palabras"]:
            drops["extensión fuera de rango"] += 1
        else:
            rows.append({**j, "text": text})
    return rows, dict(drops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--jobs", default="claude_jobs.jsonl")
    ap.add_argument("--texts", default="claude_texts.jsonl")
    args = ap.parse_args()

    rows, drops = load_jobs_and_texts(args.jobs, args.texts)
    print(f"textos válidos: {len(rows)}   descartados: {drops or 'ninguno'}\n")
    cal = json.loads(CAL_PATH.read_text())
    (w_clf, w_bino), b = cal["coef"], cal["intercept"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    p = np.clip(score_texts(clf, tok, [r["text"] for r in rows], args.device), 1e-6, 1 - 1e-6)
    del clf
    torch.cuda.empty_cache()

    bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, args.device, args.device)
    b_scores = np.array([bino.score(r["text"]) for r in rows])
    ensemble = w_clf * np.log(p / (1 - p)) + w_bino * b_scores + b

    print("Umbrales calibrados en validación:", {k: round(v, 2) for k, v in cal["thresholds"].items()}, "\n")
    groups = defaultdict(list)
    for r, e, pi in zip(rows, ensemble, p):
        groups[("TODO",)].append((e, pi))
        groups[(r["tarea"],)].append((e, pi))
        groups[(r["tarea"], r["idioma"], r["tipo"])].append((e, pi))
    for key in sorted(groups, key=lambda k: (len(k), k)):
        vals = np.array([v[0] for v in groups[key]])
        pv = np.array([v[1] for v in groups[key]])
        name = " / ".join(key)
        cells = [f"{name:<44} n={len(vals):>4}"]
        for fpr, thr in cal["thresholds"].items():
            cells.append(f"detectado@{fpr}={float((vals > thr).mean()):.2f}")
        cells.append(f"p_clf mediana={np.median(pv):.3f}")
        print("  ".join(cells))


if __name__ == "__main__":
    main()
