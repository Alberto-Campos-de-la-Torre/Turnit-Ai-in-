"""Ataque de parafraseo: ¿se evade el detector reescribiendo el texto de IA?

Es el ataque que de verdad usan los alumnos (QuillBot y similares). QuillBot no tiene
API pública y automatizar su web va contra sus términos, así que se reproduce con un
LLM local, que es lo que esas herramientas hacen por dentro: reescribir conservando el
significado. Se prueban una y dos pasadas, porque la literatura (Krishna et al., 2023)
muestra que el parafraseo recursivo es lo que tumba a los detectores.

Paso 1 (necesita Ollama):  python -m scripts.attack_parafraseo --paraphrase
Paso 2 (necesita GPU):     python -m scripts.attack_parafraseo --score
"""
import argparse
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher

import requests

from scripts.common import CORPUS

SALIDA = CORPUS / "ataque_parafraseo.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/chat"
PROMPT = {
    "es": "Reescribe el siguiente texto con otras palabras, manteniendo el significado, "
          "los datos y aproximadamente la misma extensión. Responde solo con el texto "
          "reescrito, sin comentarios:\n\n{t}",
    "en": "Rewrite the following text in different words, keeping the meaning, the data "
          "and roughly the same length. Reply with the rewritten text only, no "
          "comments:\n\n{t}",
}


def parafrasear(modelo, texto, lang):
    r = requests.post(OLLAMA, json={
        "model": modelo,
        "messages": [{"role": "user", "content": PROMPT[lang].format(t=texto)}],
        "stream": False, "think": False,
        "options": {"temperature": 0.8, "num_predict": 3072, "num_ctx": 8192},
    }, timeout=900)
    r.raise_for_status()
    return " ".join(r.json()["message"]["content"].split())


def paso_parafraseo(args):
    rows = [json.loads(l) for l in (CORPUS / "dataset.jsonl").open() if '"test"' in l]
    ia = [r for r in rows if r["label"] == "ai" and len(r["text"].split()) >= 150]
    rng = random.Random(17)
    muestra = []
    for lang in ("es", "en"):
        pool = [r for r in ia if r["lang"] == lang]
        rng.shuffle(pool)
        muestra += pool[: args.n // 2]

    hechos = {json.loads(l)["id"] for l in SALIDA.open()} if SALIDA.exists() else set()
    with SALIDA.open("a") as f:
        for i, r in enumerate(muestra, 1):
            if r["id"] in hechos:
                continue
            try:
                p1 = parafrasear(args.model, r["text"], r["lang"])
                p2 = parafrasear(args.model, p1, r["lang"])
            except Exception as e:
                print(f"  ! {r['id']}: {e}", flush=True)
                continue
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "generator": r["generator"],
                                "original": r["text"], "parafraseo_1": p1, "parafraseo_2": p2,
                                "parafraseador": args.model}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(muestra)}", flush=True)
    print(f"listo -> {SALIDA}")


def paso_puntuacion(args):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
    from scripts.calibrate import CAL_PATH
    from scripts.probe_shortcuts import score_texts
    from scripts.train_classifier import MODEL_DIR

    filas = [json.loads(l) for l in SALIDA.open()]
    cal = json.loads(CAL_PATH.read_text())
    (w_clf, w_bino), b = cal["coef"], cal["intercept"]
    thr = cal["thresholds"]["1.0%"]
    versiones = ["original", "parafraseo_1", "parafraseo_2"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    p = {v: np.clip(score_texts(clf, tok, [f[v] for f in filas], args.device), 1e-6, 1 - 1e-6)
         for v in versiones}
    del clf
    torch.cuda.empty_cache()

    bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, args.device, args.device)
    scores = {v: w_clf * np.log(p[v] / (1 - p[v]))
                 + w_bino * np.array([bino.score(f[v]) for f in filas]) + b
              for v in versiones}

    lang = np.array([f["lang"] for f in filas])
    print(f"\n{len(filas)} textos de IA parafraseados con {filas[0]['parafraseador']}; "
          f"umbral del 1% de falsos positivos\n")
    print(f"{'versión':<16}{'detectado':>11}{'detectado es':>15}{'detectado en':>15}"
          f"{'conserva del original':>24}")
    for v in versiones:
        sim = np.mean([SequenceMatcher(None, f["original"].split(), f[v].split()).ratio() for f in filas])
        det = (scores[v] > thr).mean()
        det_es = (scores[v][lang == "es"] > thr).mean()
        det_en = (scores[v][lang == "en"] > thr).mean()
        print(f"{v:<16}{det:>10.1%}{det_es:>15.1%}{det_en:>15.1%}{sim:>24.2f}")

    por_gen = defaultdict(list)
    for f, s0, s2 in zip(filas, scores["original"], scores["parafraseo_2"]):
        por_gen[f["generator"]].append((s0 > thr, s2 > thr))
    print("\npor generador original (antes -> después de 2 pasadas):")
    for g, v in sorted(por_gen.items()):
        a = np.mean([x[0] for x in v]); d = np.mean([x[1] for x in v])
        print(f"  {g:<18} n={len(v):>4}  {a:>6.1%} -> {d:>6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    if args.paraphrase:
        paso_parafraseo(args)
    if args.score:
        paso_puntuacion(args)


if __name__ == "__main__":
    main()
