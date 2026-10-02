"""Extracción de prosa de un documento real (PDF, DOCX, texto).

Un PDF académico no es prosa: trae portada, encabezados de sección, pies de figura,
tablas, fórmulas, números de página y bibliografía. Si todo eso entra al análisis, el
porcentaje marcado se diluye y un documento escrito entero con IA puede salir limpio.

Esta limpieza es la misma que se aplicó al construir el corpus (`scripts/fetch_theses.py`),
de modo que la app analiza el mismo tipo de texto con el que se entrenó y se calibró.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MIN_PALABRAS_PARRAFO = 25        # por debajo de esto no es un párrafo de prosa
PROPORCION_LETRAS = 0.75         # menos letras que esto: tabla, fórmula o listado

FIN_SECCION = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s+)?(referencias|bibliograf[íi]a|references|bibliography|"
    r"anexos?|appendix|ap[ée]ndice)\b", re.I | re.M)
PARECE_REFERENCIA = re.compile(r"(et al\.|\(\d{4}\)|doi|arxiv|https?://|pp\.|vol\.|“[^”]+,”)", re.I)
SOLO_NUMERO = re.compile(r"^[\d\s.,;:%()\[\]/-]+$")


@dataclass
class Extraccion:
    texto: str                   # prosa lista para analizar
    palabras_prosa: int
    palabras_descartadas: int
    motivo_principal: str | None


def _texto_crudo(nombre: str, datos: bytes) -> str:
    sufijo = Path(nombre).suffix.lower()
    if sufijo == ".pdf":
        import pymupdf
        with pymupdf.open(stream=datos, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    if sufijo == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(datos)).paragraphs)
    if sufijo in (".txt", ".md", ""):
        return datos.decode("utf-8", errors="replace")
    raise ValueError(f"Formato no soportado: {sufijo}. Usa .pdf, .docx o .txt.")


def extraer_prosa(nombre: str, datos: bytes) -> Extraccion:
    texto = _texto_crudo(nombre, datos)
    palabras_totales = len(texto.split())

    # Cortar bibliografía y anexos (solo si aparecen pasada la mitad del documento).
    finales = [m.start() for m in FIN_SECCION.finditer(texto) if m.start() > len(texto) * 0.5]
    if finales:
        texto = texto[: finales[0]]

    texto = re.sub(r"-\n(?=[a-záéíóúñü])", "", texto)       # palabras cortadas por guion
    texto = texto.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # Encabezados y pies que se repiten en muchas páginas (título, universidad, numeración).
    conteo = Counter(re.sub(r"\d+", "", l).strip().lower() for l in texto.splitlines())
    repetidas = {l for l, c in conteo.items() if c >= 3 and l}

    buenos, motivos = [], Counter()
    for bloque in re.split(r"\n\s*\n", texto):
        lineas = [l for l in bloque.splitlines()
                  if re.sub(r"\d+", "", l).strip().lower() not in repetidas]
        parrafo = re.sub(r"\s+", " ", " ".join(lineas)).strip()
        n = len(parrafo.split())
        if n < MIN_PALABRAS_PARRAFO:
            motivos["líneas cortas (títulos, pies, numeración)"] += n
        elif SOLO_NUMERO.match(parrafo):
            motivos["tablas o cifras"] += n
        elif sum(c.isalpha() for c in parrafo) / max(len(parrafo), 1) < PROPORCION_LETRAS:
            motivos["tablas, fórmulas o listados"] += n
        elif len(PARECE_REFERENCIA.findall(parrafo)) > 2:
            motivos["bibliografía"] += n
        elif parrafo.count(".") < n / 60:
            # Prosa tiene puntos. Sin ellos es una portada, un índice o una lista.
            motivos["portada, índice o listado"] += n
        elif sum(w.isupper() and len(w) > 2 for w in parrafo.split()) > 0.3 * len(parrafo.split()):
            motivos["texto en mayúsculas (portadas, encabezados)"] += n
        else:
            buenos.append(parrafo)

    prosa = "\n\n".join(buenos)
    descartadas = palabras_totales - len(prosa.split())
    principal = motivos.most_common(1)[0][0] if motivos else None
    return Extraccion(prosa, len(prosa.split()), max(descartadas, 0), principal)
