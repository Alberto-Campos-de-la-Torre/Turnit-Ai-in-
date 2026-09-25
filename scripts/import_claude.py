"""Integra los textos escritos por Claude al corpus, con el mismo formato que ai.jsonl.

Los de claude_texts_train_*.jsonl entran a entrenamiento; los de claude_texts.jsonl
(los 300 de la primera medición) entran como test. `finalize_corpus` se encarga después
de limpiarlos y de descartar idioma incorrecto, negativas y textos demasiado cortos.

Uso: python -m scripts.import_claude   (luego: python -m scripts.finalize_corpus)
"""
import argparse
import json
from collections import Counter

from scripts.common import CORPUS

GENERATOR = "claude"
SOURCES = [(f"claude_jobs_train_{i}.jsonl", f"claude_texts_train_{i}.jsonl") for i in range(1, 6)]
SOURCES.append(("claude_jobs.jsonl", "claude_texts.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default=GENERATOR, help="nombre del generador en el corpus")
    ap.add_argument("--pairs", nargs="*", help="pares archivo_encargos:archivo_textos")
    args = ap.parse_args()
    generator = args.generator
    sources = [tuple(x.split(":", 1)) for x in args.pairs] if args.pairs else SOURCES

    humans = {d["id"]: d for d in map(json.loads, (CORPUS / "human.jsonl").open())}
    ai_path = CORPUS / "ai.jsonl"
    existing = {json.loads(l)["id"] for l in ai_path.open()} if ai_path.exists() else set()

    rows, drops = [], Counter()
    for jobs_name, texts_name in sources:
        jobs_path, texts_path = CORPUS / jobs_name, CORPUS / texts_name
        if not texts_path.exists():
            print(f"  (falta {texts_name})")
            continue
        jobs = {j["job_id"]: j for j in map(json.loads, jobs_path.open())}
        for d in map(json.loads, texts_path.open()):
            job = jobs.get(d["job_id"])
            text = (d.get("texto") or "").strip()
            if job is None:
                drops["encargo desconocido"] += 1
                continue
            if not text:
                drops["texto vacío"] += 1
                continue
            task, human_id = d["job_id"].split(":", 1)
            h = humans.get(human_id)
            if h is None:
                drops["sin texto humano de referencia"] += 1
                continue
            row_id = f"{task}:{generator}:{human_id}"
            if row_id in existing:
                drops["ya estaba"] += 1
                continue
            rows.append({
                "id": row_id, "source_id": h["source_id"], "human_id": human_id,
                "lang": h["lang"], "type": h["type"], "kind": h["kind"], "title": h["title"],
                "section": h.get("section"), "text": text,
                "label": "ai" if task == "write" else "ai_polished",
                "generator": generator, "task": "write" if task == "write" else "polish",
                "split": h["split"],
            })

    with ai_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} textos de Claude añadidos a {ai_path.name}   descartados: {dict(drops) or 'ninguno'}")
    print(Counter((r["split"], r["label"], r["lang"], r["kind"]) for r in rows))


if __name__ == "__main__":
    main()
