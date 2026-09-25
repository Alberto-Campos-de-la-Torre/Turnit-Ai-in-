"""Análisis completo de un documento: clasificador + Binoculars + umbral calibrado.

Devuelve una puntuación global y una por ventana de oraciones, con tres veredictos:

- `ia`: por encima del umbral estricto (1% de falsos positivos en validación).
- `gris`: entre el umbral del 5% y el del 1%. No afirma nada; solo sugiere revisar.
- `humano`: por debajo.

Nada de esto es prueba de nada: es un indicio para conversar con el alumno.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars

# Dónde viven modelos y datos. Se configura con DETECTOR_IA_HOME (ver README).
BASE = Path(os.environ.get("DETECTOR_IA_HOME", Path.home() / "detector-ia-datos"))
MODEL_DIR = BASE / "models" / "mdeberta-detector"
CAL_PATH = MODEL_DIR.parent / "calibration.json"
MIN_WORDS = 150          # por debajo de esto no hay señal suficiente
MAX_LEN = 512


@dataclass
class Window:
    start: int
    end: int
    text: str
    p_clf: float
    binoculars: float
    score: float
    verdict: str
    words: int


@dataclass
class Analysis:
    words: int
    score: float
    verdict: str
    percent_ai: float        # % de palabras en ventanas marcadas como IA
    percent_gray: float
    windows: list[Window]
    warning: str | None = None

    def to_dict(self):
        d = asdict(self)
        d["windows"] = [asdict(w) if not isinstance(w, dict) else w for w in self.windows]
        return d


class Detector:
    def __init__(self, device: str = "cuda:1"):
        self.device = device
        self.cal = json.loads(CAL_PATH.read_text())
        self.w_clf, self.w_bino = self.cal["coef"]
        self.bias = self.cal["intercept"]
        self.thr_strict = self.cal["thresholds"]["1.0%"]
        self.thr_loose = self.cal["thresholds"]["5.0%"]
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.clf = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), dtype=torch.float32).to(device).eval()
        self.bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, device, device)

    @torch.inference_mode()
    def _p_ai(self, texts: list[str]) -> list[float]:
        out = []
        for i in range(0, len(texts), 8):
            enc = self.tok(texts[i : i + 8], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.clf(**enc).logits.float()
            out += torch.softmax(logits, -1)[:, 1].tolist()
        return out

    def _combine(self, p_clf: float, bino: float) -> float:
        p = min(max(p_clf, 1e-6), 1 - 1e-6)
        return self.w_clf * math.log(p / (1 - p)) + self.w_bino * bino + self.bias

    def _verdict(self, score: float) -> str:
        return "ia" if score > self.thr_strict else "gris" if score > self.thr_loose else "humano"

    def analyze(self, text: str) -> Analysis:
        text = text.strip()
        words = len(text.split())
        if words < MIN_WORDS:
            return Analysis(words, float("nan"), "insuficiente", 0.0, 0.0, [],
                            f"El texto tiene {words} palabras. Por debajo de {MIN_WORDS} "
                            "no hay señal suficiente para decir nada.")

        global_bino, segments = self.bino.score_segments(text)
        texts = [s.text for s in segments] or [text]
        probs = self._p_ai(texts)
        p_doc = self._p_ai([text])[0]

        windows = []
        for seg, p in zip(segments, probs):
            score = self._combine(p, seg.score)
            windows.append(Window(seg.start, seg.end, seg.text, p, seg.score, score,
                                  self._verdict(score), len(seg.text.split())))

        doc_score = self._combine(p_doc, global_bino)
        total = sum(w.words for w in windows) or words
        pct = lambda v: 100.0 * sum(w.words for w in windows if w.verdict == v) / total
        percent_ai, percent_gray = pct("ia"), pct("gris")

        # En un trabajo escrito a medias con IA, la puntuación del documento completo se
        # diluye, así que el veredicto global toma lo más severo entre el documento y lo
        # marcado por fragmentos. Se exige un número mínimo de fragmentos marcados, no
        # solo un porcentaje: en un texto corto (4 ventanas) un único falso positivo por
        # ventana ya sería el 25% del documento. Medido sobre 213 textos humanos de test,
        # esta regla marca como IA al 0,5% (con el porcentaje solo era el 5,2%) y sigue
        # detectando el 99% de los textos de IA.
        n_ia = sum(w.verdict == "ia" for w in windows)
        n_gray = sum(w.verdict == "gris" for w in windows)
        by_windows = ("ia" if n_ia >= 3 and percent_ai >= 25 else
                      "gris" if n_ia >= 2 and (percent_ai >= 30 or percent_ai + percent_gray >= 40)
                      else "humano")
        severity = {"humano": 0, "gris": 1, "ia": 2}
        verdict = max(self._verdict(doc_score), by_windows, key=severity.get)
        return Analysis(words, doc_score, verdict, percent_ai, percent_gray, windows)
