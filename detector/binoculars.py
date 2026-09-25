"""Detector zero-shot Binoculars (Hans et al., 2024, "Spotting LLMs with Binoculars").

puntuación = log-perplejidad(texto | performer) / entropía cruzada(observer -> performer)

Un valor bajo significa texto "demasiado predecible", típico de un LLM. El umbral
depende del par de modelos, así que se calibra con datos propios (fase 3).

A diferencia de la implementación original, aquí se guardan las contribuciones por
token, para puntuar ventanas de oraciones dentro de un documento, como hace Turnitin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_OBSERVER = "Qwen/Qwen2.5-3B"
DEFAULT_PERFORMER = "Qwen/Qwen2.5-3B-Instruct"

# Fin de oración: . ! ? … seguido de espacio y de mayúscula, dígito o signo de apertura.
_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[¿¡\"'“(\[A-ZÁÉÍÓÚÑÜ0-9])")


@dataclass
class TokenScores:
    offsets: list[tuple[int, int]]  # posición en caracteres de cada token puntuado
    nll: torch.Tensor               # -log p_performer(token)
    xent: torch.Tensor              # entropía cruzada observer -> performer en esa posición


@dataclass
class Segment:
    start: int
    end: int
    text: str
    score: float
    n_tokens: int


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


class Binoculars:
    def __init__(
        self,
        observer: str = DEFAULT_OBSERVER,
        performer: str = DEFAULT_PERFORMER,
        observer_device: str = "cuda:0",
        performer_device: str = "cuda:0",
        max_tokens: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if self.tokenizer.get_vocab() != AutoTokenizer.from_pretrained(performer).get_vocab():
            raise ValueError("Observer y performer deben compartir tokenizador.")
        load = dict(torch_dtype=torch.bfloat16)
        self.observer = AutoModelForCausalLM.from_pretrained(observer, **load).to(observer_device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, **load).to(performer_device).eval()
        self.max_tokens = max_tokens

    @torch.inference_mode()
    def token_scores(self, text: str) -> TokenScores:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        nll_parts, xent_parts, kept_offsets = [], [], []
        # Trozos de max_tokens; el primer token de cada trozo no tiene contexto y no se puntúa.
        for i in range(0, len(ids), self.max_tokens):
            chunk = torch.tensor([ids[i : i + self.max_tokens]])
            if chunk.shape[1] < 2:
                continue
            obs_logits = self.observer(chunk.to(self.observer.device)).logits[0, :-1].float()
            perf_logits = self.performer(chunk.to(self.performer.device)).logits[0, :-1].float()
            obs_logits = obs_logits.to(perf_logits.device)
            # Ambos modelos pueden tener filas de vocabulario de relleno distintas.
            v = min(obs_logits.shape[-1], perf_logits.shape[-1])
            obs_logits, perf_logits = obs_logits[:, :v], perf_logits[:, :v]
            targets = chunk[0, 1:].to(perf_logits.device)

            perf_logp = F.log_softmax(perf_logits, dim=-1)
            nll_parts.append(-perf_logp.gather(1, targets[:, None])[:, 0])
            xent_parts.append(-(F.softmax(obs_logits, dim=-1) * perf_logp).sum(-1))
            kept_offsets.extend(offsets[i + 1 : i + chunk.shape[1]])
        if not nll_parts:
            return TokenScores([], torch.empty(0), torch.empty(0))
        return TokenScores(kept_offsets, torch.cat(nll_parts).cpu(), torch.cat(xent_parts).cpu())

    def score(self, text: str) -> float:
        ts = self.token_scores(text)
        return float(ts.nll.sum() / ts.xent.sum()) if len(ts.nll) else float("nan")

    def score_segments(self, text: str, min_tokens: int = 60) -> tuple[float, list[Segment]]:
        """Puntuación global y por ventanas de oraciones consecutivas (>= min_tokens tokens).

        Cada token se puntúa con el contexto previo del documento, no solo el de su ventana.
        """
        ts = self.token_scores(text)
        if not len(ts.nll):
            return float("nan"), []
        global_score = float(ts.nll.sum() / ts.xent.sum())
        token_starts = torch.tensor([s for s, _ in ts.offsets])

        segments, win_start, win_end = [], None, None
        sentences = split_sentences(text)
        for k, (s, e) in enumerate(sentences):
            win_start = s if win_start is None else win_start
            win_end = e
            mask = (token_starts >= win_start) & (token_starts < win_end)
            is_last = k == len(sentences) - 1
            if int(mask.sum()) >= min_tokens or is_last:
                n = int(mask.sum())
                if n == 0:
                    break
                # Una cola demasiado corta se une a la ventana anterior.
                if n < min_tokens // 2 and segments:
                    prev = segments.pop()
                    win_start = prev.start
                    mask = (token_starts >= win_start) & (token_starts < win_end)
                    n = int(mask.sum())
                score = float(ts.nll[mask].sum() / ts.xent[mask].sum())
                segments.append(Segment(win_start, win_end, text[win_start:win_end], score, n))
                win_start = None
        return global_score, segments
