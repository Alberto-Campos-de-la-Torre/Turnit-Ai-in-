"""Descarga un conjunto pequeño de abstracts humanos anteriores a 2022 (es + en)
para la prueba de humo de la fase 1.

- Español: Crossref (revistas latinoamericanas/españolas de ingeniería y computación).
- Inglés: API de arXiv (cs.LG, cs.CL, cs.CV, stat.ML), fecha de envío de la v1 < 2022.

Salida: data/smoke/human.jsonl
"""
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "smoke" / "human.jsonl"
N_PER_LANG = 25
MIN_WORDS, MAX_WORDS = 110, 350

ES_QUERIES = [
    "aprendizaje automático redes neuronales algoritmo datos",
    "minería de datos clasificación algoritmo",
    "visión por computadora procesamiento de imágenes red neuronal",
    "ciencia de datos análisis predictivo modelo",
    "sistema de información desarrollo de software metodología",
]
ES_STOP = {"el", "la", "de", "que", "en", "los", "las", "del", "se", "por", "una", "para", "con"}
EN_STOP = {"the", "of", "and", "to", "in", "is", "we", "that", "for", "this", "with"}


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[#\w]+;", " ", text)
    text = re.sub(r"^\s*(Resumen|Abstract)[:.\s]*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def is_lang(text: str, stop: set, other: set) -> bool:
    words = re.findall(r"\w+", text.lower())
    return sum(w in stop for w in words) > 2 * sum(w in other for w in words)


def fetch_es() -> list[dict]:
    docs, seen = [], set()
    for q in ES_QUERIES:
        r = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": q,
                "filter": "from-pub-date:2010-01-01,until-pub-date:2021-12-31,has-abstract:true",
                "rows": 60,
                "select": "title,abstract,published,DOI",
            },
            timeout=60,
        )
        r.raise_for_status()
        for it in r.json()["message"]["items"]:
            text = clean(it.get("abstract", ""))
            n = len(text.split())
            if not (MIN_WORDS <= n <= MAX_WORDS) or not is_lang(text, ES_STOP, EN_STOP):
                continue
            # Algunos registros traen el resumen en español seguido del inglés: descartarlos.
            if re.search(r"\b(abstract|keywords)\b", text, re.I):
                continue
            key = text[:80]
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "id": f"crossref:{it['DOI']}",
                "lang": "es",
                "title": clean((it.get("title") or [""])[0]),
                "year": it["published"]["date-parts"][0][0],
                "text": text,
            })
        time.sleep(1)
    return docs


def fetch_en() -> list[dict]:
    docs = []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for cat in ["cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.SE"]:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"cat:{cat} AND submittedDate:[201501010000 TO 202112312359]",
                "start": random.randint(0, 2000),
                "max_results": 30,
            },
            timeout=60,
        )
        r.raise_for_status()
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            text = clean(e.findtext("a:summary", "", ns))
            if not (MIN_WORDS <= len(text.split()) <= MAX_WORDS):
                continue
            published = e.findtext("a:published", "", ns)  # fecha de la v1
            if not published or int(published[:4]) >= 2022:
                continue
            docs.append({
                "id": "arxiv:" + e.findtext("a:id", "", ns).rsplit("/", 1)[-1],
                "lang": "en",
                "title": clean(e.findtext("a:title", "", ns)),
                "year": int(published[:4]),
                "text": text,
            })
        time.sleep(3)  # arXiv pide >= 3 s entre llamadas
    return docs


def main():
    random.seed(0)
    es, en = fetch_es(), fetch_en()
    random.shuffle(es)
    random.shuffle(en)
    docs = es[:N_PER_LANG] + en[:N_PER_LANG]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for d in docs:
            f.write(json.dumps({**d, "label": "human", "generator": None}, ensure_ascii=False) + "\n")
    print(f"es={len(es[:N_PER_LANG])} en={len(en[:N_PER_LANG])} -> {OUT}")


if __name__ == "__main__":
    main()
