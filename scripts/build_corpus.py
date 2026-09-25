"""Une los textos humanos recolectados, los deduplica y asigna particiones por documento.

Todos los fragmentos de una misma tesis (y sus contrapartes IA) quedan en la misma
partición, para que el modelo no vea en entrenamiento el tema de un texto de prueba.

Salida: data/corpus/human.jsonl
"""
import hashlib
import json
from collections import Counter

from scripts.common import CORPUS, MAX_HUMAN_YEAR

SOURCES = ["openalex_es_article", "openalex_es_dissertation", "openalex_en_article",
           "openalex_en_dissertation", "theses_passages_es", "theses_passages_en"]


def split_of(source_id: str) -> str:
    h = int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def main():
    seen, rows = set(), []
    for name in SOURCES:
        path = CORPUS / "raw" / f"{name}.jsonl"
        if not path.exists():
            print(f"  (falta {path.name})")
            continue
        for line in path.open():
            d = json.loads(line)
            if d["hash"] in seen or not d.get("year") or d["year"] > MAX_HUMAN_YEAR:
                continue
            seen.add(d["hash"])
            d.setdefault("source_id", d["id"])
            d.setdefault("section", None)
            d.update(label="human", generator=None, task=None, split=split_of(d["source_id"]))
            d.pop("pdf_url", None)
            rows.append(d)
    out = CORPUS / "human.jsonl"
    with out.open("w") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(rows)} textos humanos -> {out}")
    for k, v in sorted(Counter((d["lang"], d["kind"], d["split"]) for d in rows).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
