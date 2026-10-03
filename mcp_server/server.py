"""Servidor MCP del detector: expone el análisis como herramientas para un asistente.

No carga el modelo: habla con la app local (`run_app.sh`), que ya lo tiene en memoria.
Así el modelo ocupa la GPU una sola vez, sin importar cuántos clientes MCP haya.

Arranque (stdio, que es lo que usan los clientes MCP):
    .venv/bin/python -m mcp_server.server

Variables: DETECTOR_URL (por defecto http://127.0.0.1:8000)
"""
import os
from pathlib import Path

import requests
from mcp.server.mcpserver import MCPServer

URL = os.environ.get("DETECTOR_URL", "http://127.0.0.1:8000").rstrip("/")
TIEMPO_LIMITE = 900

mcp = MCPServer(
    name="detector-ia",
    instructions=(
        "Detector local de texto generado con IA, en español e inglés. Marca los "
        "fragmentos que un detector señalaría, para reescribirlos antes de enviar el "
        "trabajo. Es una señal de revisión, NO una prueba de deshonestidad: un texto "
        "reformulado con una herramienta automática también se marca aunque las ideas "
        "sean propias. Nunca presentes su salida como evidencia de plagio o de fraude."
    ),
)


def _pedir(payload: dict) -> dict:
    try:
        r = requests.post(f"{URL}/api/analizar", json=payload, timeout=TIEMPO_LIMITE)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": f"No pude hablar con el detector en {URL}: {e}. "
                         "¿Está corriendo run_app.sh?"}


def _resumen(d: dict, n_fragmentos: int) -> dict:
    if "error" in d:
        return d
    fragmentos = [f for f in d["fragmentos"] if f["estado"] != "humano"][:n_fragmentos]
    return {
        "estado": d["estado"],
        "etiqueta": d["etiqueta"],
        "palabras": d["palabras"],
        "porcentaje_marcado_como_ia": d["porcentaje_ia"],
        "porcentaje_en_zona_gris": d["porcentaje_gris"],
        "puntuacion": d["puntuacion"],
        "aviso": d.get("aviso"),
        "fragmentos_a_reescribir": [
            {"estado": f["estado"], "palabras": f["palabras"], "texto": f["texto"]}
            for f in fragmentos
        ],
        "total_fragmentos_marcados": sum(1 for f in d["fragmentos"] if f["estado"] != "humano"),
        "procedencia": d.get("procedencia"),
        "nota": d["nota"],
    }


@mcp.tool(
    description=(
        "Analiza un texto y devuelve qué fragmentos marcaría un detector de IA, para "
        "reescribirlos. Necesita al menos 150 palabras. El resultado es una señal de "
        "revisión, no una prueba de deshonestidad."
    )
)
def analizar_texto(texto: str, fragmentos: int = 5, nivel: str = "normal") -> dict:
    """Analiza el texto que se le pasa directamente.

    Args:
        texto: el texto a revisar (mínimo 150 palabras).
        fragmentos: cuántos fragmentos marcados devolver, de mayor a menor puntuación.
        nivel: "normal" (1% de falsos positivos), "estricto" (5%) o "exhaustivo" (10%).
            Para revisar un texto propio conviene "estricto" o "exhaustivo": la escritura
            muy dirigida por el autor queda en zona intermedia y el nivel normal no la ve.
    """
    return _resumen(_pedir({"texto": texto, "nivel": nivel}), fragmentos)


@mcp.tool(
    description=(
        "Analiza un archivo .pdf, .docx o .txt del disco y devuelve qué fragmentos "
        "marcaría un detector de IA. Señal de revisión, no prueba de deshonestidad."
    )
)
def analizar_archivo(ruta: str, fragmentos: int = 5, nivel: str = "normal") -> dict:
    """Analiza un documento del disco.

    Args:
        ruta: ruta al archivo (.pdf, .docx, .txt).
        fragmentos: cuántos fragmentos marcados devolver.
        nivel: "normal", "estricto" o "exhaustivo" (ver analizar_texto).
    """
    return _resumen(_pedir({"ruta": str(Path(ruta).expanduser()), "nivel": nivel}),
                    fragmentos)


@mcp.tool(description="Comprueba si el detector local está disponible y responde.")
def estado() -> dict:
    """Comprueba que la app del detector esté levantada."""
    try:
        r = requests.get(f"{URL}/", timeout=10)
        return {"disponible": r.status_code == 200, "url": URL}
    except requests.RequestException as e:
        return {"disponible": False, "url": URL, "detalle": str(e),
                "sugerencia": "Levanta la app con ~/detector-ia/run_app.sh"}


if __name__ == "__main__":
    mcp.run()
