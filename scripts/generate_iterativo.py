"""Genera texto de IA escrito como se escribe de verdad: en conversación, guiado por el autor.

El corpus anterior solo tenía generación de una sola pasada ("escribe el resumen de este
título"), y por eso el detector falla con documentos reales: un trabajo escrito con IA a lo
largo de varios turnos tiene palabras de máquina pero trayectoria humana, y fragmentos de
procedencia mezclada.

La simulación usa un fragmento humano real (de una tesis del corpus) como el material que
el autor ya tiene, y encadena cuatro turnos que imitan lo que un autor pide de verdad:

1. escribir la sección a partir de unos puntos,
2. concretar un párrafo con un dato propio,
3. integrar un fragmento escrito por el autor (texto humano real, insertado tal cual),
4. recortar y quitar adjetivos.

El resultado se etiqueta `ai` con tarea `iterativo`: las palabras son de la máquina.

Uso: python -m scripts.generate_iterativo --n 400 --model gemma3:12b
"""
import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from scripts.common import CORPUS

SALIDA = CORPUS / "ai_iterativo.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/chat"


def oraciones(texto: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", texto) if len(s.split()) > 6]


def turnos(fuente: dict, rng: random.Random) -> list[str]:
    """Los cuatro mensajes del autor, construidos con material del texto humano."""
    ors = oraciones(fuente["text"])
    puntos = rng.sample(ors, min(3, len(ors)))
    dato = rng.choice(ors)
    mio = " ".join(rng.sample(ors, min(2, len(ors))))
    seccion = fuente.get("section") or "desarrollo"
    es = fuente["lang"] == "es"
    n = max(250, min(len(fuente["text"].split()), 500))

    if es:
        return [
            f'Estoy escribiendo mi tesis "{fuente["title"]}". Necesito la sección '
            f'"{seccion}", unas {n} palabras. Apóyate en estos puntos que tengo:\n'
            + "\n".join(f"- {p}" for p in puntos)
            + "\nEscribe solo el texto, en prosa continua, sin títulos ni viñetas.",
            f"Está muy general. Concreta el segundo párrafo y mete este dato que tengo: {dato}",
            f"Integra este fragmento que escribí yo, ajustando el estilo para que encaje, "
            f"sin cambiar lo que dice: {mio}",
            "Ahora recórtalo un 15%, quita los adjetivos innecesarios y las frases de relleno. "
            "Devuelve solo el texto final.",
        ]
    return [
        f'I am writing my thesis "{fuente["title"]}". I need the "{seccion}" section, about '
        f'{n} words. Base it on these points I have:\n' + "\n".join(f"- {p}" for p in puntos)
        + "\nWrite the text only, continuous prose, no headings or bullets.",
        f"Too general. Make the second paragraph concrete and include this data I have: {dato}",
        f"Integrate this fragment I wrote myself, adjusting the style so it fits, without "
        f"changing what it says: {mio}",
        "Now cut it by 15%, remove unnecessary adjectives and filler. Return the final text only.",
    ]


def limpiar(texto: str) -> str:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    texto = re.sub(r"^\s*\**(Texto final|Final text|Versión final)\**[:.]?\s*", "", texto, flags=re.I)
    texto = re.sub(r"^#+\s.*$", "", texto, flags=re.M).replace("**", "")
    return re.sub(r"\s+", " ", texto).strip()


def conversar(modelo: str, mensajes_autor: list[str], think) -> str:
    """Devuelve el texto final; si el último turno recortó demasiado, usa el anterior."""
    historial, versiones = [], []
    for mensaje in mensajes_autor:
        historial.append({"role": "user", "content": mensaje})
        cuerpo = {"model": modelo, "messages": historial, "stream": False,
                  "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 2048,
                              "num_ctx": 8192}}
        if think is not None:
            cuerpo["think"] = think
        r = requests.post(OLLAMA, json=cuerpo, timeout=900)
        r.raise_for_status()
        salida = limpiar(r.json()["message"]["content"])
        historial.append({"role": "assistant", "content": salida})
        versiones.append(salida)
    for v in reversed(versiones[-2:]):
        if len(v.split()) >= 150:
            return v
    return max(versiones, key=lambda v: len(v.split()))


GENERADORES = {"gemma3:12b": None, "mistral-nemo:12b": None, "llama3.1:8b": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--models", default="gemma3:12b,mistral-nemo:12b,llama3.1:8b")
    ap.add_argument("--hilos", type=int, default=3)
    args = ap.parse_args()

    modelos = args.models.split(",")
    rows = [json.loads(l) for l in (CORPUS / "dataset.jsonl").open() if '"train"' in l]
    fuentes = [r for r in rows if r["split"] == "train" and r["label"] == "human"
               and len(r["text"].split()) >= 200]
    rng = random.Random(31)
    rng.shuffle(fuentes)
    fuentes = fuentes[: args.n]

    hechos = {json.loads(l)["id"] for l in SALIDA.open()} if SALIDA.exists() else set()
    print(f"{len(fuentes)} conversaciones por simular", flush=True)
    tareas = []
    for i, fuente in enumerate(fuentes):
        modelo = modelos[i % len(modelos)]
        row_id = f"iterativo:{modelo}:{fuente['id']}"
        if row_id not in hechos:
            tareas.append((row_id, modelo, fuente))

    lock, cuenta = threading.Lock(), {"ok": 0, "fallos": 0}

    def trabajar(t):
        row_id, modelo, fuente = t
        try:
            texto = conversar(modelo, turnos(fuente, random.Random(fuente["id"])),
                              GENERADORES.get(modelo))
        except Exception as e:
            with lock:
                cuenta["fallos"] += 1
            print(f"  ! {fuente['id']}: {type(e).__name__}: {e}", flush=True)
            return None
        if len(texto.split()) < 120:
            with lock:
                cuenta["fallos"] += 1
            return None
        return {
            "id": row_id, "source_id": fuente["source_id"], "human_id": fuente["id"],
            "lang": fuente["lang"], "type": fuente["type"], "kind": fuente["kind"],
            "title": fuente["title"], "section": fuente.get("section"), "text": texto,
            "label": "ai", "generator": f"{modelo}-iterativo", "task": "iterativo",
            "split": "train",
        }

    with SALIDA.open("a") as f, ThreadPoolExecutor(args.hilos) as ex:
        for k, fila in enumerate(ex.map(trabajar, tareas), 1):
            if fila:
                with lock:
                    cuenta["ok"] += 1
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                    f.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(tareas)} ok={cuenta['ok']} fallos={cuenta['fallos']}", flush=True)
    print(f"listo: {cuenta['ok']} textos, {cuenta['fallos']} fallos -> {SALIDA}")


if __name__ == "__main__":
    main()
