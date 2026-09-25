"""Uso: python -m detector.cli ARCHIVO.txt  (o '-' para leer de stdin)

Muestra la puntuación Binoculars global y la de cada ventana de oraciones.
Puntuación más baja = más parecido a texto generado por IA. Todavía no hay
umbral calibrado (fase 3), así que las cifras son relativas.
"""
import argparse
import sys
import textwrap

from .binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivo")
    ap.add_argument("--observer", default=DEFAULT_OBSERVER)
    ap.add_argument("--performer", default=DEFAULT_PERFORMER)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    text = sys.stdin.read() if args.archivo == "-" else open(args.archivo, encoding="utf-8").read()
    bino = Binoculars(args.observer, args.performer, args.device, args.device)
    global_score, segments = bino.score_segments(text)

    print(f"Puntuación global: {global_score:.4f}  ({len(segments)} ventanas)\n")
    for i, seg in enumerate(segments, 1):
        preview = textwrap.shorten(seg.text, 110, placeholder=" …")
        print(f"[{i:>2}] {seg.score:.4f}  {seg.n_tokens:>4} tok  {preview}")


if __name__ == "__main__":
    main()
