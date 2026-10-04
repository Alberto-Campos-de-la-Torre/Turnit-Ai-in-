"""API de la app, con un detector falso: no carga modelos ni usa GPU."""
import io

import pytest
from fastapi.testclient import TestClient

from detector.consistencia import analizar as analizar_consistencia
from detector.pipeline import Analysis, Window

LARGO = ("La detección de anomalías en series temporales multivariadas es una tarea "
         "crítica con aplicaciones en seguridad de redes y mantenimiento predictivo. ") * 12


def _analisis(verdict="ia", p=0.99, n=6):
    ventanas = [Window(0, 10, f"fragmento {i} del documento analizado", p, 0.9,
                       3.0 if verdict == "ia" else -30.0, verdict, 150) for i in range(n)]
    return Analysis(words=900, score=3.0 if verdict == "ia" else -30.0, verdict=verdict,
                    percent_ai=100.0 if verdict == "ia" else 0.0, percent_gray=0.0,
                    windows=ventanas,
                    consistencia=analizar_consistencia([p] * n))


class DetectorFalso:
    def __init__(self):
        self.niveles = []

    def analyze(self, texto, estricto=False, nivel=None):
        self.niveles.append(nivel)
        if len(texto.split()) < 150:
            return Analysis(len(texto.split()), float("nan"), "insuficiente", 0.0, 0.0, [],
                            None, "El texto tiene pocas palabras.")
        return _analisis()


@pytest.fixture
def cliente(monkeypatch):
    from app import server
    falso = DetectorFalso()
    monkeypatch.setattr(server, "detector", falso)
    cliente = TestClient(server.app)
    cliente.falso = falso
    return cliente


def test_formulario_responde(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "Modo estricto" in r.text


def test_api_texto(cliente):
    r = cliente.post("/api/analizar", json={"texto": LARGO})
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "ia"
    assert d["etiqueta"] == "necesita reescritura"
    assert d["porcentaje_ia"] == 100.0
    assert len(d["fragmentos"]) == 6
    assert d["nota"]


def test_api_devuelve_la_procedencia(cliente):
    d = cliente.post("/api/analizar", json={"texto": LARGO}).json()
    assert d["procedencia"]["fragmentos_de_maquina"] == 6
    assert d["procedencia"]["etiqueta"] == "homogéneo, y parece de máquina"


def test_api_pasa_el_nivel(cliente):
    cliente.post("/api/analizar", json={"texto": LARGO, "nivel": "exhaustivo"})
    assert cliente.falso.niveles[-1] == "exhaustivo"


def test_api_estricto_equivale_al_nivel(cliente):
    cliente.post("/api/analizar", json={"texto": LARGO, "estricto": True})
    assert cliente.falso.niveles[-1] == "estricto"


def test_api_texto_corto_avisa(cliente):
    d = cliente.post("/api/analizar", json={"texto": "muy corto"}).json()
    assert d["estado"] == "insuficiente"
    assert d["aviso"]


def test_api_sin_texto(cliente):
    assert "error" in cliente.post("/api/analizar", json={}).json()


def test_api_archivo_inexistente(cliente):
    d = cliente.post("/api/analizar", json={"ruta": "/no/existe.pdf"}).json()
    assert "No existe" in d["error"]


def test_api_archivo(cliente, tmp_path):
    ruta = tmp_path / "trabajo.txt"
    ruta.write_text(LARGO)
    d = cliente.post("/api/analizar", json={"ruta": str(ruta)}).json()
    assert d["estado"] == "ia"
    assert d["extraccion"]["palabras_prosa"] > 100


def test_subida_por_formulario(cliente):
    r = cliente.post("/analizar", files={"archivo": ("t.txt", LARGO.encode())})
    assert r.status_code == 200
    assert "Necesita reescritura" in r.text
    assert "Qué reescribir primero" in r.text


def test_subida_de_formato_no_soportado(cliente):
    r = cliente.post("/analizar", files={"archivo": ("hoja.xlsx", b"x")})
    assert "no soportado" in r.text


def test_documento_humano_no_muestra_lista_de_reescritura(cliente, monkeypatch):
    from app import server
    monkeypatch.setattr(server.detector, "analyze",
                        lambda t, **k: _analisis("humano", 0.001))
    r = cliente.post("/analizar", data={"texto": LARGO})
    assert "Listo" in r.text
    assert "Qué reescribir primero" not in r.text
