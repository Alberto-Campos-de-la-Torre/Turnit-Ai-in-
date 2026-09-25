"""Utilidades compartidas para construir el corpus."""
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import urllib3.util.connection

# La red de esta máquina resuelve IPv6 pero no lo enruta bien: cada conexión de requests
# esperaba ~40 s antes de caer a IPv4 (curl lo evita con "happy eyeballs"). Forzar IPv4.
urllib3.util.connection.HAS_IPV6 = False

ROOT = Path(__file__).resolve().parent.parent
# El corpus vive fuera del repositorio (ocupa GB y contiene material de terceros).
# DETECTOR_IA_HOME apunta a esa carpeta; ver README.
BASE = Path(os.environ.get("DETECTOR_IA_HOME", Path.home() / "detector-ia-datos"))
CORPUS = BASE / "corpus"

# Último año aceptado como texto humano (ChatGPT salió el 30-nov-2022).
MAX_HUMAN_YEAR = 2021


def openalex_headers() -> dict:
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        raise SystemExit("Falta OPENALEX_API_KEY: ejecuta `source ~/.config/openalex.env`.")
    return {"Authorization": f"Bearer {key}"}


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[#\w]+;", " ", text)
    text = re.sub(r"^\s*(Resumen|Abstract|Summary)[:.\s]*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    norm = re.sub(r"\W+", "", text.lower())[:500]
    return hashlib.sha1(norm.encode()).hexdigest()


@lru_cache(maxsize=1)
def _lang_detector():
    from lingua import Language, LanguageDetectorBuilder
    langs = [Language.SPANISH, Language.ENGLISH, Language.PORTUGUESE,
             Language.FRENCH, Language.ITALIAN, Language.CATALAN, Language.GERMAN]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def detect_lang(text: str) -> str | None:
    """Código ISO 639-1 si la detección es clara, si no None."""
    det = _lang_detector()
    conf = det.compute_language_confidence_values(text[:2000])
    if not conf or conf[0].value < 0.8:
        return None
    return conf[0].language.iso_code_639_1.name.lower()
