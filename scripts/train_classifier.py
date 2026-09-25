"""Entrena el clasificador multilingüe (mDeBERTa-v3) humano vs IA.

Los textos `ai_polished` no entran en el entrenamiento: son un caso intermedio y
etiquetarlos como IA metería ruido. Se evalúan aparte.

Uso: python -m scripts.train_classifier [--epochs 3] [--device cuda:1]
Guarda el mejor modelo por AUROC de validación en MODEL_DIR.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.common import CORPUS

BASE_MODEL = "microsoft/mdeberta-v3-base"
BASE = Path(os.environ.get("DETECTOR_IA_HOME", Path.home() / "detector-ia-datos"))
MODEL_DIR = BASE / "models" / "mdeberta-detector"
MAX_LEN = 512


class TextDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = 0 if r["label"] == "human" else 1
        return enc


def load_rows(split, labels=("human", "ai")):
    return [d for d in map(json.loads, (CORPUS / "dataset.jsonl").open())
            if d["split"] == split and d["label"] in labels]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        mask[i, :k] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "labels": torch.tensor([b["labels"] for b in batch])}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        labels.append(batch.pop("labels").numpy())
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def report(name, probs, labels):
    auroc = roc_auc_score(labels, probs)
    human = probs[labels == 0]
    line = f"{name}: AUROC={auroc:.4f}"
    for fpr in (0.01, 0.05):
        thr = np.quantile(human, 1 - fpr)
        line += f"  TPR@{fpr:.0%}={float((probs[labels == 1] > thr).mean()):.3f}"
    print(line, flush=True)
    return auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_rows, val_rows = load_rows("train"), load_rows("val")
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    fn = lambda b: collate(b, tok.pad_token_id)
    train_loader = DataLoader(TextDataset(train_rows, tok), batch_size=args.batch_size,
                              shuffle=True, collate_fn=fn, num_workers=4)
    val_loader = DataLoader(TextDataset(val_rows, tok), batch_size=32, collate_fn=fn, num_workers=4)

    # El checkpoint de mDeBERTa está en float16 y transformers lo respeta; entrenar con
    # pesos fp16 hace que AdamW produzca NaN en el primer paso. Se cargan en float32 y el
    # cálculo se hace en bf16 con autocast.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, dtype=torch.float32).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 100 == 0:
                print(f"  época {epoch} paso {step}/{len(train_loader)} loss={running / 100:.4f}", flush=True)
                running = 0.0
        probs, labels = predict(model, val_loader, args.device)
        auroc = report(f"val época {epoch}", probs, labels)
        if auroc > best and not math.isnan(auroc):
            best = auroc
            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tok.save_pretrained(MODEL_DIR)
            print(f"  guardado (mejor AUROC={best:.4f}) -> {MODEL_DIR}", flush=True)
    print(f"Listo. Mejor AUROC de validación: {best:.4f}")


if __name__ == "__main__":
    main()
