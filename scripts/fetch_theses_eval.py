"""Descarga tesis en español que NO están en el corpus, para medir falsos positivos.

Son trabajos reales de alumnos (licenciatura y maestría), anteriores a 2022, y ninguno
participó en el entrenamiento ni en la calibración. Es el equivalente en español de la
prueba con los ensayos de PERSUADE.

Uso: python -m scripts.fetch_theses_eval --lang es --max-theses 250
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from scripts.common import CORPUS
from scripts.fetch_theses import process


def ids_en_corpus() -> set[str]:
    usados = set()
    for nombre in ("human.jsonl", "dataset.jsonl"):
        p = CORPUS / nombre
        if p.exists():
            for d in map(json.loads, p.open()):
                usados.add(d.get("source_id") or d["id"])
    return usados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="es", choices=["es", "en"])
    ap.add_argument("--max-theses", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    usados = ids_en_corpus()
    recs = [json.loads(l) for l in (CORPUS / "raw" / f"theses_pdf_{args.lang}.jsonl").open()]
    recs = list({r["id"]: r for r in recs if r["id"] not in usados}.values())
    print(f"{len(recs)} tesis candidatas que no están en el corpus")
    random.seed(123)
    random.shuffle(recs)
    per_host, selected = {}, []
    for r in recs:
        host = urlparse(r["pdf_url"]).netloc
        if per_host.get(host, 0) < 40:
            per_host[host] = per_host.get(host, 0) + 1
            selected.append(r)
    selected = selected[: args.max_theses * 4]

    out = CORPUS / f"alumnos_tesis_{args.lang}.jsonl"
    n_ok, n_pass, seen = 0, 0, set()
    with out.open("w") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(process, r) for r in selected]
        for k, fut in enumerate(as_completed(futures), 1):
            filas = [x for x in fut.result() if x["hash"] not in seen]
            if filas:
                n_ok += 1
                for x in filas:
                    seen.add(x["hash"])
                    f.write(json.dumps({"id": f"tesis:{x['id']}", "lang": x["lang"],
                                        "kind": "thesis_passage", "year": x["year"],
                                        "text": x["text"]}, ensure_ascii=False) + "\n")
                n_pass += len(filas)
            if k % 50 == 0:
                print(f"  {k}/{len(selected)} procesadas, {n_ok} tesis, {n_pass} fragmentos", flush=True)
            if n_ok >= args.max_theses:
                for fu in futures:
                    fu.cancel()
                break
    print(f"{n_ok} tesis nuevas, {n_pass} fragmentos -> {out}")


if __name__ == "__main__":
    main()
