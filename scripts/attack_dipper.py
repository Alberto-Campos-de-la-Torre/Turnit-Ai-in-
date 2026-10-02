"""Ataque de parafraseo con DIPPER, el parafraseador del artículo de Krishna et al. (2023).

DIPPER (T5-XXL, 11B, entrenado sobre PAR3) está hecho específicamente para evadir
detectores: en ese artículo hizo caer a DetectGPT del 70.3% al 4.6%. A diferencia del
ataque con gemma3:12b, este modelo NO participó en el entrenamiento del detector, así
que la prueba es limpia.

Solo inglés: DIPPER se entrenó con texto inglés.

Paso 1: python -m scripts.attack_dipper --paraphrase --n 120
Paso 2: python -m scripts.attack_dipper --score
"""
import argparse
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher

from scripts.common import CORPUS

SALIDA = CORPUS / "ataque_dipper.jsonl"            # partición test: solo para evaluar
SALIDA_TRAIN = CORPUS / "dipper_train.jsonl"       # partición train: datos adversarios
SALIDA_HUMANO = CORPUS / "dipper_humano.jsonl"     # textos humanos parafraseados: control
MODELO = "kalpeshk2011/dipper-paraphraser-xxl"
# El repositorio de DIPPER trae un tokenizador roto (vocabulario de 104 piezas) y el
# modelo responde vacío con él. Hay que usar el de T5, como indican sus autores.
TOKENIZADOR = "google/t5-v1_1-xxl"
VENTANA = 3          # oraciones por paso, como en el código original


class Dipper:
    def __init__(self, device: str = "cuda:1", dtype: str = "bf16"):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, T5ForConditionalGeneration
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(TOKENIZADOR)
        if dtype == "8bit":
            # No usar: T5 en int8 desborda y devuelve texto vacío. Se deja por registro.
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                      "device_map": {"": device}}
        else:
            # bf16 (22 GB) repartido entre las GPUs según la memoria libre de cada una,
            # para no desalojar los otros servicios de la máquina.
            libre = {}
            for i in range(torch.cuda.device_count()):
                free, _ = torch.cuda.mem_get_info(i)
                libre[i] = f"{max(int(free / 2**30) - 2, 0)}GiB"
            print(f"memoria libre por GPU: {libre}", flush=True)
            kwargs = {"dtype": torch.bfloat16, "device_map": "auto",
                      "max_memory": {**libre, "cpu": "80GiB"}}
        self.model = T5ForConditionalGeneration.from_pretrained(MODELO, **kwargs)
        self.model.eval()
        self.device = device

    def paraphrase(self, texto: str, lex: int = 60, orden: int = 60) -> str:
        """lex y orden: diversidad léxica y de orden, 0-100 (el código original usa 100-x)."""
        from detector.binoculars import split_sentences
        lex_code, orden_code = 100 - lex, 100 - orden
        texto = " ".join(texto.split())
        oraciones = [texto[a:b] for a, b in split_sentences(texto)]
        prefijo, salida = "", ""
        for i in range(0, len(oraciones), VENTANA):
            bloque = " ".join(oraciones[i : i + VENTANA])
            entrada = f"lexical = {lex_code}, order = {orden_code}"
            if prefijo:
                entrada += f" {prefijo}"
            entrada += f" <sent> {bloque} </sent>"
            enc = self.tok([entrada], return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with self.torch.inference_mode():
                out = self.model.generate(**enc, do_sample=True, top_p=0.75, top_k=None,
                                          max_length=512)
            trozo = self.tok.batch_decode(out, skip_special_tokens=True)[0]
            prefijo = (prefijo + " " + trozo).strip()[-1500:]
            salida += " " + trozo
        return " ".join(salida.split())


def paso_parafraseo(args):
    if args.label == "human":
        destino = SALIDA_HUMANO
    else:
        destino = SALIDA_TRAIN if args.split == "train" else SALIDA
    rows = [json.loads(l) for l in (CORPUS / "dataset.jsonl").open()
            if f'"{args.split}"' in l]
    ia = [r for r in rows if r["split"] == args.split and r["label"] == args.label
          and r["lang"] == "en" and len(r["text"].split()) >= 150]
    random.Random(23).shuffle(ia)
    muestra = ia[: args.n]
    hechos = {json.loads(l)["id"] for l in destino.open()} if destino.exists() else set()
    muestra = [r for r in muestra if r["id"] not in hechos]
    print(f"{len(muestra)} textos por parafrasear", flush=True)

    dipper = Dipper(args.device, args.dtype)
    with destino.open("a") as f:
        for i, r in enumerate(muestra, 1):
            try:
                p1 = dipper.paraphrase(r["text"])
                p2 = dipper.paraphrase(p1)
            except Exception as e:
                print(f"  ! {r['id']}: {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps({"id": r["id"], "lang": "en", "generator": r["generator"],
                                "label": args.label,
                                "original": r["text"], "parafraseo_1": p1, "parafraseo_2": p2,
                                "parafraseador": "dipper-xxl"}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(muestra)}", flush=True)
    print(f"listo -> {destino}")


def paso_puntuacion(args):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from detector.binoculars import DEFAULT_OBSERVER, DEFAULT_PERFORMER, Binoculars
    from scripts.calibrate import CAL_PATH
    from scripts.probe_shortcuts import score_texts
    from scripts.train_classifier import MODEL_DIR

    filas = [json.loads(l) for l in SALIDA.open()]
    cal = json.loads(CAL_PATH.read_text())
    (w_clf, w_bino), b = cal["coef"], cal["intercept"]
    thr = cal["thresholds"]["1.0%"]
    versiones = ["original", "parafraseo_1", "parafraseo_2"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32).to(args.device).eval()
    p = {v: np.clip(score_texts(clf, tok, [f[v] for f in filas], args.device), 1e-6, 1 - 1e-6)
         for v in versiones}
    del clf
    torch.cuda.empty_cache()

    bino = Binoculars(DEFAULT_OBSERVER, DEFAULT_PERFORMER, args.device, args.device)
    scores, clas = {}, {}
    for v in versiones:
        bb = np.array([bino.score(f[v]) for f in filas])
        scores[v] = w_clf * np.log(p[v] / (1 - p[v])) + w_bino * bb + b
        clas[v] = p[v]

    print(f"\n{len(filas)} textos de IA en inglés parafraseados con DIPPER-XXL "
          f"(no participó en el entrenamiento)\n")
    print(f"{'versión':<16}{'detectado':>11}{'p_clasificador':>17}{'conserva del original':>24}")
    for v in versiones:
        sim = np.mean([SequenceMatcher(None, f["original"].split(), f[v].split()).ratio()
                       for f in filas])
        print(f"{v:<16}{(scores[v] > thr).mean():>10.1%}{np.median(clas[v]):>17.4f}{sim:>24.2f}")

    por_gen = defaultdict(list)
    for f, s0, s2 in zip(filas, scores["original"], scores["parafraseo_2"]):
        por_gen[f["generator"]].append((s0 > thr, s2 > thr))
    print("\npor generador original (antes -> después de 2 pasadas):")
    for g, v in sorted(por_gen.items()):
        print(f"  {g:<18} n={len(v):>4}  {np.mean([x[0] for x in v]):>6.1%} -> "
              f"{np.mean([x[1] for x in v]):>6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--label", default="ai", choices=["ai", "human"],
                    help="human: parafrasea texto humano, como control de falsos positivos")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "8bit"])
    args = ap.parse_args()
    if args.paraphrase:
        paso_parafraseo(args)
    if args.score:
        paso_puntuacion(args)


if __name__ == "__main__":
    main()
