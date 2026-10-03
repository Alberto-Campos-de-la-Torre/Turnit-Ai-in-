"""Construye el conjunto de evaluación con documentos REALES escritos con IA.

Son trabajos del usuario (tareas, protocolo de tesis) escritos con asistencia de IA en
conversación: varios turnos, con sus instrucciones, sus datos y sus correcciones. Es el
caso que el corpus sintético no cubre y donde el detector falla.

Los documentos no se copian al repositorio: solo su prosa extraída, que vive en
$DETECTOR_IA_HOME/corpus (fuera del repositorio y fuera de git).

Uso: python -m scripts.build_eval_real --patron "Tarea*.pdf" "protocolo*.pdf" ...
"""
import argparse
import json
from pathlib import Path

from detector.extraccion import extraer_prosa
from scripts.common import CORPUS, text_hash

EXCLUIR = ("recibo", "receta", "pitch", "poster", "clase")   # personal o no es prosa propia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default=str(Path.home() / "Descargas"))
    ap.add_argument("--patrones", nargs="*", default=["Tarea*.pdf", "Proyecto*.pdf",
                                                      "protocolo*.pdf", "Fundamentos*.pdf",
                                                      "Análisis*.pdf"])
    ap.add_argument("--min-palabras", type=int, default=300)
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    rutas = sorted({r for p in args.patrones for r in carpeta.glob(p)})
    rutas = [r for r in rutas if not any(x in r.name.lower() for x in EXCLUIR)]

    filas, vistos = [], set()
    for ruta in rutas:
        try:
            e = extraer_prosa(ruta.name, ruta.read_bytes())
        except Exception as exc:
            print(f"  ! {ruta.name}: {type(exc).__name__}")
            continue
        if e.palabras_prosa < args.min_palabras:
            print(f"  - {ruta.name}: solo {e.palabras_prosa} palabras de prosa, se omite")
            continue
        h = text_hash(e.texto)
        if h in vistos:          # copias del mismo trabajo, (1).pdf, (2).pdf…
            print(f"  = {ruta.name}: duplicado de otro archivo, se omite")
            continue
        vistos.add(h)
        filas.append({"id": f"real:{ruta.stem}", "lang": "es", "kind": "trabajo_real",
                      "archivo": ruta.name, "palabras": e.palabras_prosa,
                      "palabras_descartadas": e.palabras_descartadas, "text": e.texto})
        print(f"  + {ruta.name}: {e.palabras_prosa} palabras de prosa "
              f"({e.palabras_descartadas} descartadas)")

    salida = CORPUS / "reales_ia.jsonl"
    with salida.open("w") as f:
        for d in filas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n{len(filas)} documentos -> {salida}")
    print(f"total de prosa: {sum(d['palabras'] for d in filas)} palabras")


if __name__ == "__main__":
    main()
