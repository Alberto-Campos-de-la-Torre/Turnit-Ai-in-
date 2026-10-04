"""Extracción de prosa: lo que NO es prosa tiene que quedar fuera.

Analizar el documento en bruto diluía el resultado: en un protocolo de tesis real un
tercio de las palabras eran bibliografía, tablas, cifras y portada.
"""
import io

import pytest

from detector.extraccion import extraer_prosa

PARRAFO = (
    "La detección de anomalías en series temporales multivariadas es una tarea crítica con "
    "aplicaciones que van desde la seguridad de redes hasta el mantenimiento predictivo. "
    "La naturaleza de alta dimensión de estos datos complica el problema, porque las "
    "dependencias entre canales no son estacionarias ni independientes entre sí. "
)


def _extraer(texto):
    return extraer_prosa("doc.txt", texto.encode())


def test_conserva_la_prosa():
    e = _extraer(PARRAFO)
    assert e.palabras_prosa > 40
    assert "detección de anomalías" in e.texto


def test_descarta_la_bibliografia_del_final():
    texto = PARRAFO * 3 + "\n\nReferencias\n\n" + PARRAFO
    e = _extraer(texto)
    assert e.palabras_prosa < len(texto.split())
    assert e.palabras_descartadas > 0


def test_descarta_portadas_en_mayusculas():
    portada = ("UNIVERSIDAD DE GUADALAJARA CENTRO UNIVERSITARIO DE CIENCIAS EXACTAS "
               "PROTOCOLO DE TESIS MAESTRIA EN INGENIERIA PRESENTA ALBERTO CAMPOS "
               "DIRECTOR DE TESIS CODIGO DE ALUMNO\n\n")
    e = _extraer(portada + PARRAFO * 2)
    assert "UNIVERSIDAD" not in e.texto


def test_descarta_tablas_y_cifras():
    tabla = "\n\n12.4 3.1 0.98 | 14.2 2.9 0.97 | 11.8 3.3 0.95 | 13.0 3.0 0.96\n\n"
    e = _extraer(PARRAFO * 2 + tabla)
    assert "0.98" not in e.texto


def test_descarta_lineas_sueltas_cortas():
    e = _extraer("3. Metodología\n\n" + PARRAFO * 2 + "\n\nFigura 4\n\n")
    assert "Figura 4" not in e.texto


def test_informa_del_motivo_principal():
    e = _extraer(PARRAFO * 2 + "\n\n" + "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20\n\n")
    assert e.motivo_principal


def test_formato_no_soportado():
    with pytest.raises(ValueError, match="no soportado"):
        extraer_prosa("hoja.xlsx", b"datos")


VARIANTES = [
    " El primer experimento se repitió en tres condiciones distintas de iluminación.",
    " La segunda medición empleó sensores inerciales calibrados en laboratorio.",
    " El tercer ensayo incorporó una validación cruzada por sujeto para evitar fugas.",
    " La última comparación añadió un grupo de control sin intervención alguna.",
]


def test_descarta_encabezados_que_se_repiten_en_cada_pagina():
    """Una línea que aparece tres veces o más es encabezado de página, no prosa."""
    encabezado = "Universidad de Guadalajara — Protocolo de tesis\n"
    texto = "".join(encabezado + PARRAFO + v + "\n\n" for v in VARIANTES)
    e = _extraer(texto)
    assert "Universidad de Guadalajara" not in e.texto
    assert "sensores inerciales" in e.texto


def test_los_encabezados_se_cuentan_sin_los_numeros_de_pagina():
    """"Página 3" y "Página 4" son el mismo encabezado: se comparan sin dígitos."""
    texto = "".join(f"Protocolo de tesis — página {i}\n" + PARRAFO + v + "\n\n"
                    for i, v in enumerate(VARIANTES))
    e = _extraer(texto)
    assert "página" not in e.texto.lower()
    assert "validación cruzada" in e.texto


def test_docx():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    for v in VARIANTES[:3]:
        doc.add_paragraph(PARRAFO + v)
    buf = io.BytesIO()
    doc.save(buf)
    e = extraer_prosa("trabajo.docx", buf.getvalue())
    assert e.palabras_prosa > 100


def test_pdf():
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    pagina = pdf.new_page()
    cuerpo = " ".join(PARRAFO + v for v in VARIANTES[:3])
    pagina.insert_textbox(pymupdf.Rect(50, 50, 545, 790), cuerpo, fontsize=10)
    e = extraer_prosa("trabajo.pdf", pdf.tobytes())
    assert e.palabras_prosa > 80
