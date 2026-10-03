"""Extrae prosa académica de las transcripciones de conversación del usuario.

Es el material más valioso del proyecto: texto escrito por Claude en conversación real,
con las correcciones del autor entre medias. Reproduce el caso donde el detector falla
(documentos reales, 9% marcado) y que las simulaciones no lograron imitar.

Filtros de privacidad, con autorización explícita del usuario (protocolo de tesis y
tareas):

- Solo bloques de prosa del asistente: nada de mensajes del usuario, código, comandos,
  rutas, salidas de terminal ni listas.
- Se descarta cualquier bloque con correos, claves, tokens, URLs o rutas del sistema.
- Nada sale de la máquina: la salida vive en $DETECTOR_IA_HOME/corpus, fuera del
  repositorio y excluida de git.

Uso: python -m scripts.extraer_transcripciones --archivos <ruta.jsonl> [...]
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from scripts.common import CORPUS, detect_lang, text_hash

SALIDA = CORPUS / "reales_transcripciones.jsonl"
MIN_PALABRAS = 120

# Señales de que un bloque no es prosa académica.
NO_PROSA = re.compile(
    r"```|^\s*[\$#>]\s|\b(pip|npm|git|sudo|cd|ls|grep|python3?|bash|curl|def |class |import |"
    r"SELECT |FROM )\b|/home/|/mnt/|\.py\b|\.json\b|\.sh\b|localhost|127\.0\.0\.1", re.I | re.M)
SENSIBLE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|sk-[A-Za-z0-9]{8}|https?://|\b\d{10,}\b|API[_ ]?KEY", re.I)
VINETAS = re.compile(r"^\s*([-*•]|\d+\.)\s", re.M)


def es_prosa(texto: str) -> bool:
    palabras = texto.split()
    if not MIN_PALABRAS <= len(palabras) <= 1200:
        return False
    if NO_PROSA.search(texto) or SENSIBLE.search(texto):
        return False
    if "|" in texto or "**" in texto or "```" in texto:   # tablas o markdown: no es prosa
        return False
    if len(VINETAS.findall(texto)) > 2:          # listas: no es prosa continua
        return False
    letras = sum(c.isalpha() or c.isspace() for c in texto)
    if letras / len(texto) < 0.92:
        return False
    if texto.count(".") < len(palabras) / 60:    # sin puntos: títulos o tablas
        return False
    return detect_lang(texto) in ("es", "en")


DOCUMENTOS = (".tex", ".md", ".txt", ".bib")
CODIGO = (".py", ".sh", ".json", ".yml", ".yaml", ".js", ".ts", ".toml", ".cfg", ".ipynb")


def limpiar_marcado(texto: str) -> str:
    """Quita LaTeX y markdown para quedarse con la prosa."""
    texto = re.sub(r"%.*$", "", texto, flags=re.M)                       # comentarios LaTeX
    texto = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", texto)      # entornos
    texto = re.sub(r"\$[^$]{0,200}\$|\\\[[^\]]{0,400}\\\]", " ", texto)      # matemáticas
    texto = re.sub(r"\\(cite|ref|label|includegraphics|input|usepackage)\s*(\[[^\]]*\])?\{[^}]*\}",
                   " ", texto)
    texto = re.sub(r"\\(section|subsection|subsubsection|chapter|title|author)\*?\{([^}]*)\}",
                   " ", texto)
    texto = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", texto)   # resto de comandos
    texto = re.sub(r"^#+\s.*$", " ", texto, flags=re.M)                 # títulos markdown
    texto = re.sub(r"\|[^\n]*\|", " ", texto)                            # tablas markdown
    texto = texto.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", texto).strip()


def bloques_asistente(ruta: Path):
    """Prosa escrita EN ARCHIVOS de documento (Write/Edit), que es donde vive el texto.

    En una sesión de Claude Code el texto del documento no va en el mensaje, va en la
    herramienta que escribe el archivo. Las ediciones además dan el antes y el después.
    """
    for linea in ruta.open(errors="replace"):
        try:
            d = json.loads(linea)
        except Exception:
            continue
        msg = d.get("message") or {}
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for bloque in msg["content"]:
            if not (isinstance(bloque, dict) and bloque.get("type") == "tool_use"):
                continue
            if bloque.get("name") not in ("Write", "Edit", "MultiEdit"):
                continue
            entrada = bloque.get("input") or {}
            ruta_archivo = str(entrada.get("file_path", "")).lower()
            if not ruta_archivo.endswith(DOCUMENTOS) or ruta_archivo.endswith(CODIGO):
                continue
            for campo in ("content", "new_string"):
                if entrada.get(campo):
                    yield limpiar_marcado(str(entrada[campo]))
            for edicion in entrada.get("edits") or []:
                if isinstance(edicion, dict) and edicion.get("new_string"):
                    yield limpiar_marcado(str(edicion["new_string"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivos", nargs="+", required=True)
    ap.add_argument("--max-por-archivo", type=int, default=400)
    args = ap.parse_args()

    filas, vistos = [], set()
    for i, nombre in enumerate(args.archivos, 1):
        ruta = Path(nombre).expanduser()
        etiqueta = f"conv{i}"            # no se guarda la ruta real
        n_archivo = 0
        previos = []
        for texto in bloques_asistente(ruta):
            texto = re.sub(r"\s+", " ", texto).strip()
            if not es_prosa(texto):
                continue
            h = text_hash(texto)
            if h in vistos:
                continue
            vistos.add(h)
            # ¿Es una revisión de un bloque anterior del mismo documento?
            revision_de = None
            for j, anterior in previos[-12:]:
                if SequenceMatcher(None, anterior.split()[:150], texto.split()[:150]).ratio() > 0.35:
                    revision_de = j
                    break
            idx = len(filas)
            filas.append({"id": f"transcripcion:{etiqueta}:{n_archivo}", "origen": etiqueta,
                          "orden": n_archivo, "lang": detect_lang(texto),
                          "palabras": len(texto.split()),
                          "revision_de": revision_de, "text": texto})
            previos.append((idx, texto))
            n_archivo += 1
            if n_archivo >= args.max_por_archivo:
                break
        print(f"{etiqueta}: {n_archivo} bloques de prosa ({ruta.name[:12]}…)")

    with SALIDA.open("w") as f:
        for d in filas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    revisiones = sum(1 for d in filas if d["revision_de"] is not None)
    print(f"\n{len(filas)} bloques -> {SALIDA}")
    print(f"  de ellos, {revisiones} son revisión de un bloque anterior (cadenas borrador→final)")
    print(f"  palabras totales: {sum(d['palabras'] for d in filas)}")
    print(f"  idiomas: {({l: sum(1 for d in filas if d['lang'] == l) for l in ('es', 'en')})}")


if __name__ == "__main__":
    main()
