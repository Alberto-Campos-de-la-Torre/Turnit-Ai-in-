"""Añade al corpus los textos de escritura iterativa (ai_iterativo.jsonl).

Las filas ya traen el esquema completo; esto solo las anexa a ai.jsonl evitando
duplicados. Después hay que correr finalize_corpus, train_classifier y calibrate.
"""
import json
from collections import Counter

from scripts.common import CORPUS


def main():
    origen, destino = CORPUS / "ai_iterativo.jsonl", CORPUS / "ai.jsonl"
    existentes = {json.loads(l)["id"] for l in destino.open()}
    nuevas, drops = [], Counter()
    for d in map(json.loads, origen.open()):
        if d["id"] in existentes:
            drops["ya estaba"] += 1
        elif len(d["text"].split()) < 120:
            drops["demasiado corto"] += 1
        else:
            nuevas.append(d)
            existentes.add(d["id"])
    with destino.open("a") as f:
        for d in nuevas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(nuevas)} textos iterativos añadidos   descartados: {dict(drops) or 'ninguno'}")
    print(Counter(d["generator"] for d in nuevas))


if __name__ == "__main__":
    main()
