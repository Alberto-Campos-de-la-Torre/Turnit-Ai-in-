"""Evalúa la app a nivel de documento completo, que es como se usará de verdad.

Arma documentos largos juntando fragmentos de una misma tesis (partición test):

- `humano`: solo fragmentos humanos -> mide falsos positivos por trabajo.
- `ia`: solo fragmentos de IA.
- `mixto-30` / `mixto-60`: ese porcentaje de palabras escritas por IA, intercaladas,
  que es el caso realista del alumno que copia una parte.

Uso: python -m scripts.eval_documents [--n 40] [--device cuda:1]
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from detector.pipeline import Detector
from scripts.common import CORPUS

MIN_WORDS = 1200


def build_docs(n: int, rng: random.Random):
    rows = [d for d in map(json.loads, (CORPUS / "dataset.jsonl").open())
            if d["split"] == "test" and d["kind"] == "thesis_passage"]
    by_source = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_source[r["source_id"]][r["label"]].append(r)
    # Tesis con suficiente material humano y de IA para armar documentos comparables.
    sources = [s for s, v in by_source.items() if len(v["human"]) >= 3 and len(v["ai"]) >= 3]
    rng.shuffle(sources)

    docs = []
    for src in sources:
        human = by_source[src]["human"]
        ia = by_source[src]["ai"]
        lang = human[0]["lang"]
        def join(parts):
            return " ".join(p["text"] for p in parts)
        full_h, full_i = join(human), join(ia)
        if len(full_h.split()) < MIN_WORDS or len(full_i.split()) < MIN_WORDS:
            continue
        k = len(human)
        mix30 = [human[i] if i % 3 else ia[i] for i in range(min(k, len(ia)))]
        mix60 = [ia[i] if i % 3 else human[i] for i in range(min(k, len(ia)))]
        docs += [{"tipo": "humano", "lang": lang, "text": full_h},
                 {"tipo": "ia", "lang": lang, "text": full_i},
                 {"tipo": "mixto-30", "lang": lang, "text": join(mix30)},
                 {"tipo": "mixto-60", "lang": lang, "text": join(mix60)}]
        if len(docs) >= 4 * n:
            break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    docs = build_docs(args.n, random.Random(11))
    det = Detector(args.device)
    stats = defaultdict(Counter)
    pct = defaultdict(list)
    for i, d in enumerate(docs, 1):
        a = det.analyze(d["text"])
        stats[d["tipo"]][a.verdict] += 1
        pct[d["tipo"]].append(a.percent_ai)
        if i % 20 == 0:
            print(f"  {i}/{len(docs)}", flush=True)

    print(f"\nDocumentos completos ({len(docs)//4} tesis, {len(docs)} documentos, "
          f"mínimo {MIN_WORDS} palabras)\n")
    print(f"{'tipo':<10} {'n':>4}  {'ia':>5} {'gris':>5} {'humano':>7}   % del texto marcado (mediana)")
    for tipo in ("humano", "mixto-30", "mixto-60", "ia"):
        c = stats[tipo]
        n = sum(c.values())
        med = sorted(pct[tipo])[len(pct[tipo]) // 2] if pct[tipo] else 0
        print(f"{tipo:<10} {n:>4}  {c['ia']:>5} {c['gris']:>5} {c['humano']:>7}   {med:.0f}%")


if __name__ == "__main__":
    main()
