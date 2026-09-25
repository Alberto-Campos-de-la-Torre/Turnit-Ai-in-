"""Genera la contraparte IA del corpus humano con varios LLM locales (Ollama).

Tareas:
- write:  el LLM escribe desde cero un texto con el mismo título (y sección, si es un
          fragmento de tesis), idioma y longitud que el humano. Etiqueta "ai".
- polish: el LLM reescribe el texto humano "mejorando la redacción". Etiqueta
          "ai_polished": es el caso ambiguo del alumno que corrige su texto con IA.

Cada texto humano recibe una tarea write de un generador aleatorio, y POLISH_FRACTION
de ellos además una polish. mistral-nemo solo se usa en la partición test, para medir
si el detector generaliza a un modelo que nunca vio.

Es reanudable: se salta lo que ya está en data/corpus/ai.jsonl.
"""
import argparse
import json
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from scripts.common import CORPUS

# nombre -> (servidor Ollama, valor de "think")
GENERATORS = {
    "gpt-oss:20b": ("http://127.0.0.1:11434", "low"),
    "qwen3:14b": ("http://127.0.0.1:11434", False),
    "qwen3:32b": ("http://127.0.0.1:11434", False),
    "llama3.1:8b": ("http://127.0.0.1:11435", None),
    "gemma3:12b": ("http://127.0.0.1:11435", None),
    "mistral-nemo:12b": ("http://127.0.0.1:11435", None),
}
HELD_OUT = "mistral-nemo:12b"
# gpt-oss pesa más porque es el más parecido a ChatGPT y el más difícil de detectar.
WEIGHTS = {"gpt-oss:20b": 3, "qwen3:14b": 1, "qwen3:32b": 1, "llama3.1:8b": 1, "gemma3:12b": 2}
POLISH_FRACTION = 0.2

PROMPTS = {
    ("write", "abstract", "es"): [
        "Escribe el resumen de un artículo académico titulado \"{title}\". Extensión aproximada: {n} palabras. "
        "Responde solo con el texto del resumen, en un único párrafo, sin título ni formato markdown.",
        "Redacta el abstract, en español, de un trabajo de investigación llamado \"{title}\" (unas {n} palabras). "
        "Devuelve únicamente el párrafo, sin encabezados.",
    ],
    ("write", "abstract", "en"): [
        "Write the abstract of an academic paper titled \"{title}\". Length: about {n} words. "
        "Reply with the abstract text only, as a single paragraph, with no title or markdown.",
        "Draft a ~{n}-word abstract for a research paper called \"{title}\". Output only the paragraph.",
    ],
    ("write", "thesis_passage", "es"): [
        "Estás escribiendo tu tesis titulada \"{title}\". Escribe un fragmento de la sección \"{section}\" "
        "de unas {n} palabras. Solo prosa continua en párrafos, sin títulos, listas ni markdown.",
        "Redacta, para la sección \"{section}\" de la tesis \"{title}\", un texto académico de aproximadamente "
        "{n} palabras. Devuelve solo los párrafos, sin encabezados ni viñetas.",
    ],
    ("write", "thesis_passage", "en"): [
        "You are writing your thesis titled \"{title}\". Write a passage of about {n} words for the section "
        "\"{section}\". Continuous prose paragraphs only, no headings, lists or markdown.",
        "Draft roughly {n} words of academic prose for the \"{section}\" section of the thesis \"{title}\". "
        "Return only the paragraphs.",
    ],
    ("polish", None, "es"): [
        "Mejora la redacción del siguiente texto académico, manteniendo su contenido y extensión. "
        "Responde solo con el texto reescrito:\n\n{text}",
        "Reescribe este texto para que suene más claro y profesional, sin cambiar las ideas. "
        "Devuelve únicamente el texto:\n\n{text}",
    ],
    ("polish", None, "en"): [
        "Improve the writing of the following academic text, keeping its content and length. "
        "Reply with the rewritten text only:\n\n{text}",
        "Rewrite this text so it reads more clearly and professionally without changing the ideas. "
        "Return only the text:\n\n{text}",
    ],
}


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*\**(Resumen|Abstract|Texto reescrito|Rewritten text)\**[:.]?\s*", "", text, flags=re.I)
    text = re.sub(r"^#+\s.*$", "", text, flags=re.M)  # encabezados markdown
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def plan_jobs(humans: list[dict]) -> list[dict]:
    names = list(WEIGHTS)
    weights = [WEIGHTS[n] for n in names]
    jobs = []
    for h in humans:
        # Semilla por documento: agregar textos al corpus no cambia lo ya planificado.
        rng = random.Random(h["id"])
        pool = names + [HELD_OUT] if h["split"] == "test" else names
        w = weights + [3] if h["split"] == "test" else weights
        tasks = ["write"] + (["polish"] if rng.random() < POLISH_FRACTION else [])
        for task in tasks:
            jobs.append({"human": h, "task": task, "generator": rng.choices(pool, w)[0],
                         "variant": rng.randrange(2), "temperature": round(rng.uniform(0.6, 1.0), 2)})
    return jobs


def run_job(job: dict) -> dict | None:
    h, task, gen = job["human"], job["task"], job["generator"]
    server, think = GENERATORS[gen]
    kind = h["kind"] if task == "write" else None
    prompt = PROMPTS[(task, kind, h["lang"])][job["variant"]].format(
        title=h["title"], section=h.get("section") or "", n=len(h["text"].split()), text=h["text"])
    body = {"model": gen, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": job["temperature"], "top_p": 0.95, "num_predict": 3072, "num_ctx": 4096}}
    if think is not None:
        body["think"] = think
    r = requests.post(f"{server}/api/chat", json=body, timeout=900)
    r.raise_for_status()
    text = clean_output(r.json()["message"]["content"])
    if len(text.split()) < 0.4 * len(h["text"].split()):
        return None  # salida truncada o negativa
    return {
        "id": f"{task}:{gen}:{h['id']}", "source_id": h["source_id"], "human_id": h["id"],
        "lang": h["lang"], "type": h["type"], "kind": h["kind"], "title": h["title"],
        "section": h.get("section"), "text": text, "label": "ai" if task == "write" else "ai_polished",
        "generator": gen, "task": task, "split": h["split"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="solo los primeros N trabajos (prueba)")
    ap.add_argument("--generators", help="lista separada por comas; por defecto todos")
    ap.add_argument("--parallel", type=int, default=4, help="peticiones simultáneas por servidor")
    args = ap.parse_args()

    humans = [json.loads(l) for l in (CORPUS / "human.jsonl").open()]
    jobs = plan_jobs(humans)
    out_path = CORPUS / "ai.jsonl"
    done = {json.loads(l)["id"] for l in out_path.open()} if out_path.exists() else set()
    # Contrapartes hechas con una versión anterior del texto humano: se regeneran y, en
    # finalize_corpus, la fila más reciente con el mismo id reemplaza a la vieja.
    stale_path = CORPUS / "stale_ai_ids.txt"
    if stale_path.exists():
        done -= set(stale_path.read_text().split())
    jobs = [j for j in jobs if f"{j['task']}:{j['generator']}:{j['human']['id']}" not in done]
    if args.generators:
        keep = set(args.generators.split(","))
        jobs = [j for j in jobs if j["generator"] in keep]
    jobs = jobs[: args.limit] if args.limit else jobs
    print(f"Trabajos pendientes: {len(jobs)}  {dict(Counter(j['generator'] for j in jobs))}", flush=True)

    lock, stats = threading.Lock(), Counter()
    with out_path.open("a") as f:
        # Un pool por servidor; dentro de cada uno se va generador por generador para no
        # obligar a Ollama a cargar y descargar modelos todo el tiempo.
        def worker_for(server):
            server_jobs = sorted((j for j in jobs if GENERATORS[j["generator"]][0] == server),
                                 key=lambda j: j["generator"])
            with ThreadPoolExecutor(args.parallel) as ex:
                for job, res in zip(server_jobs, ex.map(safe_run, server_jobs)):
                    with lock:
                        stats["ok" if res else "fail"] += 1
                        if res:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                            f.flush()
                        total = stats["ok"] + stats["fail"]
                        if total % 100 == 0:
                            print(f"  {total}/{len(jobs)} ok={stats['ok']} fallos={stats['fail']} "
                                  f"({job['generator']})", flush=True)

        servers = sorted({GENERATORS[j["generator"]][0] for j in jobs})
        threads = [threading.Thread(target=worker_for, args=(s,)) for s in servers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(f"Listo: ok={stats['ok']} fallos={stats['fail']} -> {out_path}")


def safe_run(job):
    try:
        return run_job(job)
    except Exception as e:  # un fallo de red o de Ollama no debe detener todo
        print(f"  ! {job['generator']} {job['human']['id']}: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()
