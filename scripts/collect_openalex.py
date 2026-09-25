"""Recolecta resúmenes humanos anteriores a 2022 desde OpenAlex (es + en).

- Temas: aprendizaje automático, IA, ciencia de datos, minería de datos, visión por
  computadora, PLN e ingeniería de software.
- Muestra aleatoria (`sample`) y solo trabajos con < 50 citas: los textos muy citados
  tienen más probabilidad de estar memorizados por los LLM y sesgarían la calibración.
- El idioma se verifica con lingua, porque la etiqueta de OpenAlex a veces es errónea.
- También guarda la URL del PDF de las tesis de acceso abierto, para fetch_theses.py.

Salida: data/corpus/raw/openalex_{lang}_{type}.jsonl
"""
import argparse
import json
import time

import requests

from scripts.common import CORPUS, MAX_HUMAN_YEAR, clean, detect_lang, openalex_headers, text_hash

CONCEPTS = "|".join([
    "C119857082",   # machine learning
    "C154945302",   # artificial intelligence
    "C2522767166",  # data science
    "C124101348",   # data mining
    "C31972630",    # computer vision
    "C204321447",   # natural language processing
    "C115903868",   # software engineering
])
TARGETS = {("es", "article"): 3000, ("es", "dissertation"): 2500,
           ("en", "article"): 3000, ("en", "dissertation"): 1500}
MIN_WORDS, MAX_WORDS = 100, 450


def rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def iter_works(filt: str, headers: dict):
    """Recorre la población completa si cabe en 10 000 trabajos; si no, una muestra aleatoria."""
    base = {"filter": filt, "per-page": 200,
            "select": "id,display_name,publication_year,abstract_inverted_index,"
                      "best_oa_location,cited_by_count,primary_location"}

    def get(params):
        for _ in range(5):
            r = requests.get("https://api.openalex.org/works", headers=headers, timeout=90, params=params)
            if r.status_code != 429:
                r.raise_for_status()
                return r.json()
            time.sleep(30)
        r.raise_for_status()

    count = get({"filter": filt, "per-page": 1, "select": "id"})["meta"]["count"]
    if count <= 10000:
        cursor = "*"
        while cursor:
            data = get({**base, "cursor": cursor})
            yield from data["results"]
            cursor = data["meta"].get("next_cursor") if data["results"] else None
    else:
        for page in range(1, 10000 // 200 + 1):
            data = get({**base, "sample": 10000, "seed": 42, "page": page})
            if not data["results"]:
                break
            yield from data["results"]


def collect(lang: str, wtype: str, target: int, headers: dict):
    out = CORPUS / "raw" / f"openalex_{lang}_{wtype}.jsonl"
    pdf_out = CORPUS / "raw" / f"theses_pdf_{lang}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, n_pdf, n_seen = set(), 0, 0, 0
    filt = (f"language:{lang},type:{wtype},publication_year:2008-{MAX_HUMAN_YEAR},"
            f"has_abstract:true,cited_by_count:<50,concepts.id:{CONCEPTS}")
    pdf_f = pdf_out.open("w") if wtype == "dissertation" else None
    with out.open("w") as f:
        for w in iter_works(filt, headers):
            n_seen += 1
            if n_seen % 1000 == 0:
                print(f"  {lang}/{wtype}: vistos {n_seen}, guardados {kept}, pdfs {n_pdf}", flush=True)
            oa = w.get("best_oa_location") or {}
            # El cuerpo de una tesis está en el idioma de OpenAlex aunque el resumen no lo esté.
            if pdf_f and oa.get("pdf_url"):
                pdf_f.write(json.dumps({
                    "id": w["id"].rsplit("/", 1)[-1], "lang": lang, "title": clean(w.get("display_name") or ""),
                    "year": w["publication_year"], "pdf_url": oa["pdf_url"],
                }, ensure_ascii=False) + "\n")
                n_pdf += 1
            if kept >= target:
                if pdf_f:
                    continue  # seguir solo para juntar PDFs
                break
            text = clean(rebuild_abstract(w.get("abstract_inverted_index")))
            if not MIN_WORDS <= len(text.split()) <= MAX_WORDS or detect_lang(text) != lang:
                continue
            h = text_hash(text)
            if h in seen:
                continue
            seen.add(h)
            src = (w.get("primary_location") or {}).get("source") or {}
            f.write(json.dumps({
                "id": w["id"].rsplit("/", 1)[-1], "lang": lang, "type": wtype,
                "kind": "abstract", "title": clean(w.get("display_name") or ""),
                "year": w["publication_year"], "venue": src.get("display_name"),
                "cited_by": w.get("cited_by_count"), "pdf_url": oa.get("pdf_url"),
                "text": text, "hash": h,
            }, ensure_ascii=False) + "\n")
            kept += 1
    if pdf_f:
        pdf_f.close()
    print(f"{lang}/{wtype}: {kept} resúmenes -> {out}" + (f"; {n_pdf} PDFs -> {pdf_out}" if pdf_f else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="p. ej. es:dissertation")
    args = ap.parse_args()
    headers = openalex_headers()
    for (lang, wtype), target in TARGETS.items():
        if args.only and args.only != f"{lang}:{wtype}":
            continue
        collect(lang, wtype, target, headers)


if __name__ == "__main__":
    main()
