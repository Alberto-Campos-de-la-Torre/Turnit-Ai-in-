"""Mide el detector sobre los documentos REALES escritos con IA (reales_ia.jsonl).

Es la prueba que de verdad importa: trabajos del usuario escritos con asistencia de IA
en conversación. Un detector entrenado solo con generación de una sola pasada marca aquí
muy poco (línea base: 9% de media, 5 de 11 documentos como "listo").

Uso: python -m scripts.eval_reales [--device cuda:1]
"""
import argparse
import json

from detector.pipeline import Detector
from scripts.common import CORPUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    docs = [json.loads(l) for l in (CORPUS / "reales_ia.jsonl").open()]
    det = Detector(args.device)
    print(f"{'documento':<42}{'palabras':>9}{'% IA':>7}{'% gris':>8}  estado")
    ia = gris = listos = 0
    for d in docs:
        a = det.analyze(d["text"])
        ia += a.percent_ai
        gris += a.percent_gray
        listos += a.verdict == "humano"
        print(f"{d['archivo'][:40]:<42}{d['palabras']:>9}{a.percent_ai:>7.0f}"
              f"{a.percent_gray:>8.0f}  {a.verdict}")
    n = len(docs)
    print(f"\npromedio: {ia/n:.0f}% marcado como IA, {gris/n:.0f}% en zona gris")
    print(f"documentos que salen como 'listo' (fallo total): {listos} de {n}")


if __name__ == "__main__":
    main()
