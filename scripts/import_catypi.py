"""Extrae los textos del corpus CATyPI (TEI XML) para medir falsos positivos.

CATyPI son secciones de tesis y propuestas de investigación en computación y
tecnologías de la información, en español, de licenciatura, maestría y doctorado
(INAOE, CC BY-NC-SA 4.0). El archivo es de 2018, así que todo es escritura humana.

Es el corpus más parecido a los alumnos del usuario que hemos conseguido.

Salida: data/corpus/alumnos_catypi.jsonl
"""
import json
import re
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.common import CORPUS

BASE = Path(os.environ.get("DETECTOR_IA_HOME", Path.home() / "detector-ia-datos"))
FUENTE = BASE / "externo" / "catypi" / "CATyPI_corpus_TEI.xml"
NS = {"t": "http://www.tei-c.org/ns/1.0"}
NIVEL = {"undergraduate": "licenciatura", "master": "maestria", "doctoral": "doctorado"}


def texto_de(seccion) -> str:
    """Une los párrafos de una sección, quitando encabezados y marcas de referencias."""
    partes = []
    for p in seccion.iter(f"{{{NS['t']}}}p"):
        partes.append(" ".join(p.itertext()))
    texto = " ".join(partes)
    texto = re.sub(r"\bREFS\b", " ", texto)
    texto = re.sub(r"\s+([,.;:])", r"\1", texto)
    return re.sub(r"\s+", " ", texto).strip()


def main():
    root = ET.parse(FUENTE).getroot()
    filas, stats = [], Counter()
    for i, tesis in enumerate(root.find(".//t:body", NS).findall("t:div", NS)):
        titulo = (tesis.findtext("t:head/t:title", "", NS) or "").strip()
        nivel = NIVEL.get(tesis.get("subtype"), tesis.get("subtype"))
        for j, seccion in enumerate(tesis.findall("t:div", NS)):
            texto = texto_de(seccion)
            palabras = len(texto.split())
            stats[f"{palabras // 100 * 100}-{palabras // 100 * 100 + 99} palabras"] += 1
            if palabras < 60:
                stats["descartada por corta"] += 1
                continue
            filas.append({
                "id": f"catypi:{i}-{j}", "lang": "es", "kind": "thesis_section",
                "nivel": nivel, "seccion": seccion.get("subtype"),
                "titulo": titulo, "palabras": palabras, "text": texto,
            })

    out = CORPUS / "alumnos_catypi.jsonl"
    with out.open("w") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(filas)} secciones -> {out}")
    print("por nivel:", dict(Counter(r["nivel"] for r in filas)))
    print("por sección:", dict(Counter(r["seccion"] for r in filas)))
    print("longitudes:", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
