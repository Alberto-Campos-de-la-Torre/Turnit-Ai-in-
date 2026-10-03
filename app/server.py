"""App web local para analizar un trabajo. Solo escucha en 127.0.0.1.

Arranque:  ~/detector-ia/run_app.sh      (luego abrir http://127.0.0.1:8000)
"""
import html
import io
import textwrap
import re
from pathlib import Path

import pymupdf
from docx import Document
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from detector.extraccion import extraer_prosa
from detector.pipeline import MIN_WORDS, Detector

# El aviso por idioma dejó de hacer falta al entrenar con textos de Claude: en inglés
# la detección pasó del 5% al 100% (textos escritos por Claude, partición de prueba).

app = FastAPI(title="Detector local de texto con IA")
detector: Detector | None = None

TEMPLATE = Path(__file__).parent / "page.html"


def get_detector() -> Detector:
    global detector
    if detector is None:
        detector = Detector()
    return detector


def extract_text(name: str, data: bytes):
    """Devuelve solo la prosa del documento (sin bibliografía, tablas ni encabezados).

    Analizar el documento en bruto diluía el resultado: en un protocolo de tesis real,
    un tercio de las palabras eran referencias, cifras y títulos, y eso bajaba el
    porcentaje marcado a menos de la mitad de lo que corresponde.
    """
    e = extraer_prosa(name, data)
    return e.texto, e


def render(body: str) -> str:
    return TEMPLATE.read_text().replace("<!--CONTENT-->", body)


def render_form(message: str = "") -> str:
    aviso = f'<p class="error">{html.escape(message)}</p>' if message else ""
    return render(f"""
      {aviso}
      <form method="post" action="/analizar" enctype="multipart/form-data" class="card">
        <label for="texto">Pega el texto del trabajo</label>
        <textarea id="texto" name="texto" rows="12" placeholder="Mínimo {MIN_WORDS} palabras…"></textarea>
        <p class="sep">o sube un archivo (.pdf, .docx, .txt)</p>
        <input type="file" name="archivo" accept=".pdf,.docx,.txt,.md">
        <p class="sep"><label class="linea"><input type="checkbox" name="estricto" value="1">
          Modo estricto: cuenta también los fragmentos dudosos (para revisar antes de enviar)
        </label></p>
        <button type="submit">Analizar</button>
      </form>""")


def render_result(a, source: str) -> str:
    if a.verdict == "insuficiente":
        return render_form(a.warning)

    badge = {"ia": ("alto", "Necesita reescritura antes de enviarlo"),
             "gris": ("gris", "Conviene revisar los fragmentos marcados"),
             "humano": ("bajo", "Listo: sin fragmentos marcados como IA")}[a.verdict]

    marks = []
    for w in a.windows:
        cls = {"ia": "m-ia", "gris": "m-gris", "humano": "m-humano"}[w.verdict]
        marks.append(
            f'<span class="{cls}" title="puntuación {w.score:.2f} · '
            f'clasificador {w.p_clf:.3f} · binoculars {w.binoculars:.3f}">{html.escape(w.text)}</span>')

    por_reescribir = sorted([w for w in a.windows if w.verdict != "humano"],
                            key=lambda w: -w.score)[:5]
    if por_reescribir:
        items = "".join(
            f'<li><span class="peso">{w.words} palabras</span> '
            f'{html.escape(textwrap.shorten(w.text, 220, placeholder=" …"))}</li>'
            for w in por_reescribir)
        pendientes = f"""
      <div class="card">
        <h2>Qué reescribir primero</h2>
        <p class="sub">Ordenados por cuánto pesan en el resultado. Reescríbelos con tus
        propias palabras y vuelve a analizar: el porcentaje debería bajar.</p>
        <ol class="pendientes">{items}</ol>
      </div>"""
    else:
        pendientes = ""

    filas = "".join(
        f"<tr><td>{i}</td><td>{w.words}</td><td>{w.score:.2f}</td><td>{w.p_clf:.3f}</td>"
        f"<td>{w.binoculars:.3f}</td><td class='v-{w.verdict}'>{w.verdict}</td></tr>"
        for i, w in enumerate(a.windows, 1))

    return render(f"""
      <div class="card resumen">
        <div class="badge b-{badge[0]}">{badge[1]}</div>
        <div class="cifras">
          <div><strong>{a.percent_ai:.0f}%</strong><span>del texto con indicios altos</span></div>
          <div><strong>{a.percent_gray:.0f}%</strong><span>en zona gris</span></div>
          <div><strong>{a.words}</strong><span>palabras · {len(a.windows)} fragmentos</span></div>
          <div><strong>{a.score:.2f}</strong><span>puntuación global</span></div>
        </div>
        <p class="fuente">Fuente: {html.escape(source)}</p>
      </div>

      {pendientes}

      <div class="card">
        <h2>Texto con los fragmentos marcados</h2>
        <p class="leyenda"><span class="m-ia">indicios altos</span>
           <span class="m-gris">zona gris</span>
           <span class="m-humano">sin indicios</span></p>
        <div class="texto">{" ".join(marks)}</div>
      </div>

      <details class="card">
        <summary>Detalle por fragmento</summary>
        <table><thead><tr><th>#</th><th>Palabras</th><th>Puntuación</th>
          <th>Clasificador</th><th>Binoculars</th><th>Veredicto</th></tr></thead>
          <tbody>{filas}</tbody></table>
      </details>

      <div class="acciones">
        <a class="boton" href="/">Analizar otro</a>
        <button class="boton" onclick="window.print()">Guardar como PDF</button>
      </div>""")


@app.post("/api/analizar")
async def api_analizar(payload: dict):
    """API JSON para clientes como el servidor MCP. {"texto": "..."} o {"ruta": "..."}."""
    texto = (payload.get("texto") or "").strip()
    fuente, extraccion = "texto", None
    if not texto and payload.get("ruta"):
        ruta = Path(payload["ruta"]).expanduser()
        if not ruta.is_file():
            return {"error": f"No existe el archivo: {ruta}"}
        try:
            texto, extr = extract_text(ruta.name, ruta.read_bytes())
        except ValueError as e:
            return {"error": str(e)}
        fuente = str(ruta)
        extraccion = {"palabras_prosa": extr.palabras_prosa,
                      "palabras_descartadas": extr.palabras_descartadas,
                      "descartado_principalmente": extr.motivo_principal}
    if not texto:
        return {"error": "Hace falta 'texto' o 'ruta'."}

    a = get_detector().analyze(texto, nivel=payload.get("nivel") or
                               ("estricto" if payload.get("estricto") else "normal"))
    etiqueta = {"ia": "necesita reescritura", "gris": "conviene revisar",
                "humano": "listo", "insuficiente": "texto demasiado corto"}[a.verdict]
    return {
        "fuente": fuente, "palabras": a.words, "extraccion": extraccion, "estado": a.verdict, "etiqueta": etiqueta,
        "puntuacion": None if a.verdict == "insuficiente" else round(a.score, 3),
        "porcentaje_ia": round(a.percent_ai, 1), "porcentaje_gris": round(a.percent_gray, 1),
        "aviso": a.warning,
        "fragmentos": [
            {"estado": w.verdict, "palabras": w.words, "puntuacion": round(w.score, 3),
             "inicio": w.start, "fin": w.end, "texto": w.text}
            for w in sorted(a.windows, key=lambda w: -w.score)
        ],
        "nota": ("Señal de revisión, no prueba de deshonestidad. Un texto reformulado con "
                 "una herramienta automática también se marca."),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return render_form()


@app.post("/analizar", response_class=HTMLResponse)
async def analizar(texto: str = Form(""), archivo: UploadFile | None = File(None),
                   estricto: str = Form("")):
    source = "texto pegado"
    if archivo is not None and archivo.filename:
        data = await archivo.read()
        try:
            texto, extr = extract_text(archivo.filename, data)
        except ValueError as e:
            return render_form(str(e))
        source = (f"{archivo.filename} · {extr.palabras_prosa} palabras de prosa analizadas, "
                  f"{extr.palabras_descartadas} descartadas"
                  + (f" ({extr.motivo_principal})" if extr.motivo_principal else ""))
    if not texto.strip():
        return render_form("No recibí ningún texto.")
    return render_result(get_detector().analyze(texto, estricto=bool(estricto)), source)
