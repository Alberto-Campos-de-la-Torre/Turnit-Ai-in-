"""Genera la contraparte IA de cada abstract humano (mismo título, idioma y longitud)
con modelos locales servidos por Ollama.

Entrada: data/smoke/human.jsonl  ->  Salida: data/smoke/ai.jsonl
"""
import json
import re
import sys
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data" / "smoke"
OLLAMA = "http://127.0.0.1:11434/api/chat"
# (modelo, valor de "think"): gpt-oss no permite desactivar el razonamiento, solo bajarlo.
GENERATORS = [("gpt-oss:20b", "low"), ("qwen3:14b", False)]

PROMPTS = {
    "es": "Escribe el resumen (abstract) de un artículo académico titulado \"{title}\". "
          "Extensión aproximada: {n} palabras. Responde solo con el texto del resumen, "
          "en un único párrafo, sin título, encabezados ni formato markdown.",
    "en": "Write the abstract of an academic paper titled \"{title}\". "
          "Length: about {n} words. Reply with the abstract text only, as a single "
          "paragraph, with no title, headings or markdown formatting.",
}


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*\**(Resumen|Abstract)\**[:.]?\s*", "", text, flags=re.I)
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def generate(model: str, think, prompt: str) -> str:
    r = requests.post(OLLAMA, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 2048},
    }, timeout=600)
    r.raise_for_status()
    return clean(r.json()["message"]["content"])


def main():
    humans = [json.loads(l) for l in (DATA / "human.jsonl").open()]
    out_path = DATA / "ai.jsonl"
    done = set()
    if out_path.exists():
        done = {(d["source_id"], d["generator"]) for d in map(json.loads, out_path.open())}
    with out_path.open("a") as f:
        for model, think in GENERATORS:
            for h in humans:
                if (h["id"], model) in done:
                    continue
                prompt = PROMPTS[h["lang"]].format(title=h["title"], n=len(h["text"].split()))
                text = generate(model, think, prompt)
                if len(text.split()) < 60:
                    print(f"  ! salida corta, se omite: {model} {h['id']}", file=sys.stderr)
                    continue
                f.write(json.dumps({
                    "id": f"{model}:{h['id']}", "source_id": h["id"], "lang": h["lang"],
                    "title": h["title"], "text": text, "label": "ai", "generator": model,
                }, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{model} {h['lang']} {len(text.split())} palabras")


if __name__ == "__main__":
    main()
