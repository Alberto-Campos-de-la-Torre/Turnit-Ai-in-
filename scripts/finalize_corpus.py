"""Limpia la contraparte IA y une todo en el dataset de la fase 3.

- Quita preámbulos ("Here's the abstract:", "Aquí tienes…") y marcas markdown sueltas.
- Descarta negativas del modelo y textos IA en otro idioma o demasiado cortos.
- Descarta contrapartes cuyo texto humano ya no está en human.jsonl.
- Si una contraparte se regeneró (stale_ai_ids.txt), se queda la versión más reciente.

Salida: data/corpus/dataset.jsonl (human + ai + ai_polished) y un resumen por consola.
"""
import json
import re
from collections import Counter

from scripts.common import CORPUS, detect_lang

PREAMBLE = re.compile(
    r"^(aquí (tienes|está)|here is|here's|(claro|por supuesto|sure|certainly)[,!])[^:]{0,120}:\s*", re.I)
# Negativas del modelo (Llama se niega a veces a "escribir tu tesis"). Solo al inicio del texto.
REFUSAL = re.compile(
    r"^\W*(lo siento|lamentablemente,? no puedo|no puedo (ayudar|cumplir|generar|escribir|proporcionar)|"
    r"I('m| am) sorry|unfortunately,? I can|I can(no|')t (help|assist|write|generate|provide|fulfill)|"
    r"I('m| am) (unable|not able))", re.I)
LABEL_PREFIX = re.compile(r"^\s*(Resumen|Abstract|Texto reescrito|Rewritten text)[:.]?\s+", re.I)


def clean_ai(text: str) -> str:
    text = PREAMBLE.sub("", text)
    text = LABEL_PREFIX.sub("", text)
    text = re.sub(r"(?<!\w)[*_]{1,2}([^*_\n]+?)[*_]{1,2}(?!\w)", r"\1", text)  # *cursiva*, __negrita__
    return re.sub(r"\s+", " ", text).strip()


def main():
    humans = {d["id"]: d for d in map(json.loads, (CORPUS / "human.jsonl").open())}
    drops, ai_rows = Counter(), []
    # Si un id aparece varias veces (contraparte regenerada), vale la última.
    latest = {r["id"]: r for r in map(json.loads, (CORPUS / "ai.jsonl").open())}
    for r in latest.values():
        h = humans.get(r["human_id"])
        if h is None:
            drops["sin humano"] += 1
            continue
        r["text"] = clean_ai(r["text"])
        if REFUSAL.search(r["text"]):
            drops[f"negativa ({r['generator']})"] += 1
        elif len(r["text"].split()) < 0.4 * len(h["text"].split()):
            drops["corto"] += 1
        elif detect_lang(r["text"]) != r["lang"]:
            drops[f"idioma ({r['generator']})"] += 1
        else:
            ai_rows.append(r)

    out = CORPUS / "dataset.jsonl"
    keys = ["id", "source_id", "lang", "type", "kind", "title", "section", "text",
            "label", "generator", "task", "split"]
    with out.open("w") as f:
        for d in list(humans.values()) + ai_rows:
            f.write(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False) + "\n")

    rows = list(humans.values()) + ai_rows
    print(f"{len(rows)} textos -> {out}")
    print("descartados:", dict(drops))
    print("\npor etiqueta y partición:")
    for k, v in sorted(Counter((d["label"], d["split"]) for d in rows).items()):
        print(f"  {k}: {v}")
    print("\npor idioma, tipo y etiqueta:")
    for k, v in sorted(Counter((d["lang"], d["kind"], d["label"]) for d in rows).items()):
        print(f"  {k}: {v}")
    print("\nIA por generador:", dict(Counter(d["generator"] for d in ai_rows)))


if __name__ == "__main__":
    main()
