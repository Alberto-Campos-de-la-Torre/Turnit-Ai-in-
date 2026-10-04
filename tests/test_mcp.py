"""Servidor MCP: habla con la app por HTTP, así que se prueba con la red simulada."""
import json

import pytest


class RespuestaFalsa:
    def __init__(self, datos, status=200):
        self._datos, self.status_code = datos, status

    def json(self):
        return self._datos

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http")


RESPUESTA = {
    "fuente": "texto", "palabras": 900, "estado": "ia", "etiqueta": "necesita reescritura",
    "puntuacion": 3.0, "porcentaje_ia": 60.0, "porcentaje_gris": 10.0, "aviso": None,
    "extraccion": None,
    "fragmentos": [
        {"estado": "ia", "palabras": 150, "puntuacion": 3.5, "inicio": 0, "fin": 1, "texto": "a"},
        {"estado": "gris", "palabras": 150, "puntuacion": 0.5, "inicio": 1, "fin": 2, "texto": "b"},
        {"estado": "humano", "palabras": 150, "puntuacion": -9.0, "inicio": 2, "fin": 3, "texto": "c"},
    ],
    "procedencia": {"etiqueta": "parece mezcla de procedencias", "parece_mezcla": True,
                    "fragmentos_de_maquina": 2, "racha_mas_larga": 2,
                    "fragmentos_claramente_humanos": 3, "fragmentos_totales": 6,
                    "nota": "Indicio secundario."},
    "nota": "Señal de revisión, no prueba de deshonestidad.",
}


@pytest.fixture
def espia(monkeypatch):
    from mcp_server import server
    enviado = {}

    def post_falso(url, json=None, timeout=None):
        enviado["url"], enviado["json"] = url, json
        return RespuestaFalsa(RESPUESTA)

    monkeypatch.setattr(server.requests, "post", post_falso)
    return enviado


def test_analizar_texto_resume(espia):
    from mcp_server.server import analizar_texto
    d = analizar_texto("un texto largo", fragmentos=2)
    assert d["estado"] == "ia"
    assert d["porcentaje_marcado_como_ia"] == 60.0
    assert len(d["fragmentos_a_reescribir"]) == 2        # respeta el límite pedido
    assert d["total_fragmentos_marcados"] == 2           # ia + gris, no el humano
    assert d["procedencia"]["parece_mezcla"] is True
    assert "no prueba de deshonestidad" in d["nota"]


def test_analizar_texto_pasa_el_nivel(espia):
    from mcp_server.server import analizar_texto
    analizar_texto("un texto largo", nivel="exhaustivo")
    assert espia["json"]["nivel"] == "exhaustivo"


def test_analizar_archivo_expande_la_ruta(espia):
    from mcp_server.server import analizar_archivo
    analizar_archivo("~/trabajo.pdf")
    assert espia["json"]["ruta"].startswith("/")
    assert "~" not in espia["json"]["ruta"]


def test_si_la_app_no_responde_lo_dice(monkeypatch):
    from mcp_server import server

    def post_roto(url, json=None, timeout=None):
        raise server.requests.ConnectionError("sin conexión")

    monkeypatch.setattr(server.requests, "post", post_roto)
    d = server.analizar_texto("un texto")
    assert "error" in d
    assert "run_app.sh" in d["error"]


def test_estado_detecta_que_la_app_esta_caida(monkeypatch):
    from mcp_server import server

    def get_roto(url, timeout=None):
        raise server.requests.ConnectionError("sin conexión")

    monkeypatch.setattr(server.requests, "get", get_roto)
    d = server.estado()
    assert d["disponible"] is False
    assert "run_app.sh" in d["sugerencia"]


def test_las_instrucciones_advierten_del_uso(espia):
    from mcp_server.server import mcp
    texto = (mcp.instructions or "").lower()
    assert "no una prueba" in texto or "no es una prueba" in texto
    assert "plagio" in texto
