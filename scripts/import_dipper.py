"""Añade al corpus los textos de IA parafraseados con DIPPER (entrenamiento adversario).

Solo entran los de la partición de entrenamiento (`dipper_train.jsonl`). Los 119 de la
partición de prueba (`ataque_dipper.jsonl`) se quedan fuera a propósito: son la medida
honesta de si el entrenamiento adversario sirvió.

Cada texto aporta dos ejemplos: una pasada y dos pasadas de parafraseo. Siguen siendo
texto de IA, así que la etiqueta es `ai`.

Uso: python -m scripts.import_dipper   (luego: finalize_corpus, train_classifier, calibrate)
"""
import json
from collections import Counter

from scripts.common import CORPUS

GENERADOR = "dipper"


def main():
    humans = {d["id"]: d for d in map(json.loads, (CORPUS / "human.jsonl").open())}
    ai_path = CORPUS / "ai.jsonl"
    existentes = {json.loads(l)["id"] for l in ai_path.open()}
    origen = CORPUS / "dipper_train.jsonl"

    filas, drops = [], Counter()
    for d in map(json.loads, origen.open()):
        human_id = d["id"].rsplit(":", 1)[-1]      # el id del texto de IA acaba en el id humano
        h = humans.get(human_id)
        if h is None:
            drops["sin texto humano de referencia"] += 1
            continue
        if h["split"] != "train":
            drops["no es de entrenamiento"] += 1
            continue
        for pasadas in (1, 2):
            texto = d[f"parafraseo_{pasadas}"].strip()
            if len(texto.split()) < 100:
                drops["demasiado corto"] += 1
                continue
            row_id = f"dipper{pasadas}:{d['generator']}:{human_id}"
            if row_id in existentes:
                drops["ya estaba"] += 1
                continue
            filas.append({
                "id": row_id, "source_id": h["source_id"], "human_id": human_id,
                "lang": h["lang"], "type": h["type"], "kind": h["kind"], "title": h["title"],
                "section": h.get("section"), "text": texto, "label": "ai",
                "generator": GENERADOR, "task": f"paraphrase_{pasadas}", "split": "train",
            })

    with ai_path.open("a") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(filas)} textos parafraseados añadidos   descartados: {dict(drops) or 'ninguno'}")
    print(Counter(r["task"] for r in filas))


if __name__ == "__main__":
    main()
