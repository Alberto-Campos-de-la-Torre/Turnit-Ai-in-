"""Puntúa una partición del dataset con el clasificador entrenado.

Salida: data/corpus/scores_clf_{split}.jsonl con la probabilidad de que sea IA.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.common import CORPUS
from scripts.train_classifier import MODEL_DIR, TextDataset, collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--model", default=str(MODEL_DIR))
    args = ap.parse_args()

    rows = [d for d in map(json.loads, (CORPUS / "dataset.jsonl").open()) if d["split"] == args.split]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.float32).to(args.device).eval()
    loader = DataLoader(TextDataset(rows, tok), batch_size=32, num_workers=4,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    out = CORPUS / f"scores_clf_{args.split}.jsonl"
    with out.open("w") as f, torch.inference_mode():
        i = 0
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.float()
            for p in torch.softmax(logits, -1)[:, 1].tolist():
                r = rows[i]
                f.write(json.dumps({"id": r["id"], "label": r["label"], "p_ai": p}) + "\n")
                i += 1
            if i % 1000 < 32:
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"{i} textos -> {out}")


if __name__ == "__main__":
    main()
