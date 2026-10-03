"""Mide los falsos positivos sobre escritura de alumnos reales.

Hasta ahora la tasa de falsos positivos se midió con artículos y tesis publicados,
que están pulidos y revisados. Un alumno escribe peor y más irregular. Este script
pasa por la app completa (ventanas + ensamble + umbral calibrado) un corpus externo
de textos escritos por alumnos antes de 2022 y cuenta cuántos se marcarían.

Uso: python -m scripts.eval_alumnos --file alumnos_persuade.jsonl [--n 1200]
"""
import argparse
import json
from collections import Counter, defaultdict

from detector.pipeline import Detector
from scripts.common import CORPUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="alumnos_persuade.jsonl")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--estricto", action="store_true",
                    help="cuenta también la zona gris, como el modo estricto de la app")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (CORPUS / args.file).open()][: args.n]
    det = Detector(args.device)
    veredictos, por_grupo, marcados = Counter(), defaultdict(Counter), []
    for i, r in enumerate(rows, 1):
        a = det.analyze(r["text"], estricto=args.estricto)
        veredictos[a.verdict] += 1
        for campo in ("nota", "grado", "nivel", "seccion"):
            if r.get(campo) is not None:
                por_grupo[(campo, r[campo])][a.verdict] += 1
        if a.verdict != "humano":
            marcados.append((r["id"], a.verdict, round(a.score, 2), round(a.percent_ai)))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    n = sum(veredictos.values())
    print(f"\n{n} textos de alumnos reales (escritos antes de 2022)\n")
    for v in ("ia", "gris", "humano", "insuficiente"):
        if veredictos[v]:
            print(f"  {v:<13} {veredictos[v]:>5}  {veredictos[v]/n:>6.1%}")
    print("\npor grupo (solo % marcado como ia o gris):")
    for clave in sorted(por_grupo, key=lambda k: (k[0], str(k[1]))):
        c = por_grupo[clave]
        total = sum(c.values())
        if total >= 20:
            print(f"  {clave[0]} {clave[1]:<6} n={total:>5}  ia={c['ia']/total:>6.1%}  gris={c['gris']/total:>6.1%}")
    if marcados:
        print("\nprimeros marcados:", marcados[:8])


if __name__ == "__main__":
    main()
