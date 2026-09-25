"""Descarga tesis de acceso abierto (lista de collect_openalex.py) y extrae fragmentos de prosa.

Por cada tesis se guardan hasta PASSAGES_PER_THESIS fragmentos de 200-450 palabras,
repartidos entre el inicio, el medio y el final, con el título de la sección en la que
aparecen (se usa después para pedirle al LLM "la misma sección").

Se descartan: referencias, índices, tablas o fórmulas (poca proporción de letras),
párrafos en otro idioma y PDFs cuya fecha de creación sea diciembre de 2022 o posterior.
Los PDFs no se guardan en disco.

Salida: data/corpus/raw/theses_passages_{lang}.jsonl
"""
import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pymupdf
import requests

from scripts.common import CORPUS, clean, detect_lang, text_hash

PASSAGES_PER_THESIS = 4
MIN_WORDS, MAX_WORDS = 200, 450
MAX_PDF_BYTES = 60 * 1024 * 1024
UA = "Mozilla/5.0 (X11; Linux x86_64) detector-ia-corpus/0.1 (uso academico)"

HEADING = re.compile(
    r"^\s*((cap[íi]tulo|chapter)\s+\w+.*|([1-9]\d?(\.\d{1,2}){0,3}\.?|[IVX]+\.)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]{2,}[^\n]{0,80})\s*$"
    r"|^\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s,:]{4,80}$", re.M)
# Títulos que no son de sección: pies de figura/tabla, fuentes, URLs.
NOT_HEADING = re.compile(r"fuente|source|figura|figure|tabla|table|gr[áa]fic|http|www\.", re.I)
# Secciones cuya prosa no sirve: portada, metadatos del repositorio, índices y dedicatorias.
SKIP_SECTIONS = re.compile(r"description|afiliaci|resumen|abstract|[íi]ndice|contents|agradecimiento|"
                           r"acknowledg|dedicatoria|dedication|glosario|glossary|lista de|list of|palabras clave|keywords", re.I)
END_SECTIONS = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?(referencias|bibliograf[íi]a|references|bibliography|anexos?|appendix)\b",
                          re.I | re.M)
REF_LIKE = re.compile(r"(et al\.|\(\d{4}\)|doi|http|pp\.|vol\.)", re.I)


def pdf_is_pre_chatgpt(doc) -> bool:
    date = (doc.metadata or {}).get("creationDate") or ""
    m = re.match(r"D:(\d{4})(\d{2})", date)
    if not m:
        return True  # sin fecha: nos basamos en el año de publicación de OpenAlex
    year, month = int(m[1]), int(m[2])
    return (year, month) < (2022, 12)


def good_paragraph(p: str, lang: str) -> bool:
    words = p.split()
    if len(words) < 40:
        return False
    letters = sum(c.isalpha() for c in p)
    if letters / max(len(p), 1) < 0.75:
        return False
    if len(REF_LIKE.findall(p)) > 3:
        return False
    return detect_lang(p) == lang


def extract_passages(doc, lang: str) -> list[dict]:
    # Las primeras páginas son portada, metadatos del repositorio y resumen.
    text = "\n".join(page.get_text() for i, page in enumerate(doc) if i >= 3)
    # Cortar referencias y anexos (solo si aparecen en la segunda mitad).
    ends = [m.start() for m in END_SECTIONS.finditer(text) if m.start() > len(text) * 0.5]
    if ends:
        text = text[: ends[0]]
    text = re.sub(r"-\n(?=[a-záéíóúñ])", "", text)  # palabras cortadas por guion

    # Párrafos: bloques separados por líneas en blanco o por un punto al final de línea.
    blocks = re.split(r"\n\s*\n|(?<=[.:])\n(?=[A-ZÁÉÍÓÚÑ¿¡])", text)
    # Encabezados de página (título de la tesis, universidad...): se repiten en muchas páginas.
    line_counts = Counter(re.sub(r"\d+", "", l).strip().lower() for l in text.splitlines())
    running = {l for l, c in line_counts.items() if c >= 3 and l}

    heading, paragraphs = "Introducción" if lang == "es" else "Introduction", []
    for block in blocks:
        hm = HEADING.search(block)
        if hm and len(block.split()) < 15:
            cand = clean(hm.group(0))[:80]
            if (cand and not NOT_HEADING.search(cand)
                    and re.sub(r"\d+", "", cand).strip().lower() not in running):
                heading = cand
            continue
        if SKIP_SECTIONS.search(heading):
            continue
        p = clean(block.replace("\n", " "))
        if good_paragraph(p, lang):
            paragraphs.append((heading, p))

    # Unir párrafos consecutivos de la misma sección hasta MIN..MAX palabras.
    passages, cur, cur_head = [], [], None
    for head, p in paragraphs:
        if cur and (head != cur_head or len(" ".join(cur + [p]).split()) > MAX_WORDS):
            if len(" ".join(cur).split()) >= MIN_WORDS:
                passages.append((cur_head, " ".join(cur)))
            cur = []
        if not cur:
            cur_head = head
        if len(p.split()) <= MAX_WORDS:
            cur.append(p)
    if cur and len(" ".join(cur).split()) >= MIN_WORDS:
        passages.append((cur_head, " ".join(cur)))

    if len(passages) <= PASSAGES_PER_THESIS:
        chosen = passages
    else:  # repartidos a lo largo del documento
        step = len(passages) / PASSAGES_PER_THESIS
        chosen = [passages[int(i * step + step / 2)] for i in range(PASSAGES_PER_THESIS)]
    return [{"section": h, "text": t} for h, t in chosen]


def process(rec: dict) -> list[dict]:
    try:
        r = requests.get(rec["pdf_url"], headers={"User-Agent": UA}, timeout=(10, 60), stream=True)
        if r.status_code != 200:
            return []
        data = b""
        for chunk in r.iter_content(1 << 16):
            data += chunk
            if len(data) > MAX_PDF_BYTES:
                return []
        if not data.startswith(b"%PDF"):
            return []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count < 20 or not pdf_is_pre_chatgpt(doc):
                return []
            passages = extract_passages(doc, rec["lang"])
    except Exception:
        return []
    return [{
        "id": f"{rec['id']}#p{i}", "source_id": rec["id"], "lang": rec["lang"], "type": "dissertation",
        "kind": "thesis_passage", "title": rec["title"], "section": p["section"], "year": rec["year"],
        "text": p["text"], "hash": text_hash(p["text"]),
    } for i, p in enumerate(passages)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["es", "en"])
    ap.add_argument("--max-theses", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    recs = [json.loads(l) for l in (CORPUS / "raw" / f"theses_pdf_{args.lang}.jsonl").open()]
    recs = list({r["id"]: r for r in recs}.values())
    random.seed(0)
    random.shuffle(recs)
    # Evitar que un solo repositorio domine: máximo 60 tesis por dominio.
    per_host, selected = {}, []
    for r in recs:
        host = urlparse(r["pdf_url"]).netloc
        if per_host.get(host, 0) < 60:
            per_host[host] = per_host.get(host, 0) + 1
            selected.append(r)
    selected = selected[: int(args.max_theses * 1.8)]  # margen: muchas descargas fallan

    out = CORPUS / "raw" / f"theses_passages_{args.lang}.jsonl"
    n_ok, n_pass, seen = 0, 0, set()
    with out.open("w") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(process, r) for r in selected]
        for k, fut in enumerate(as_completed(futures), 1):
            rows = [row for row in fut.result() if row["hash"] not in seen]
            if rows:
                n_ok += 1
                for row in rows:
                    seen.add(row["hash"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_pass += len(rows)
            if k % 50 == 0:
                print(f"  {args.lang}: procesadas {k}/{len(selected)}, tesis útiles {n_ok}, fragmentos {n_pass}",
                      flush=True)
            if n_ok >= args.max_theses:
                for fu in futures:
                    fu.cancel()
                break
    print(f"{args.lang}: {n_ok} tesis, {n_pass} fragmentos -> {out}")


if __name__ == "__main__":
    main()
