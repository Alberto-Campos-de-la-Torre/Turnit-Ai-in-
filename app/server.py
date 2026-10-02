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


def extract_text(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(p.get_text() for p in doc)
        text = re.sub(r"-\n(?=[a-záéíóúñ])", "", text)
    elif suffix == ".docx":
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    elif suffix in (".txt", ".md", ""):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa .pdf, .docx o .txt.")
    return re.sub(r"[ \t]+", " ", text).strip()


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


@app.get("/", response_class=HTMLResponse)
def home():
    return render_form()


@app.post("/analizar", response_class=HTMLResponse)
async def analizar(texto: str = Form(""), archivo: UploadFile | None = File(None)):
    source = "texto pegado"
    if archivo is not None and archivo.filename:
        data = await archivo.read()
        try:
            texto = extract_text(archivo.filename, data)
        except ValueError as e:
            return render_form(str(e))
        source = archivo.filename
    if not texto.strip():
        return render_form("No recibí ningún texto.")
    return render_result(get_detector().analyze(texto), source)
