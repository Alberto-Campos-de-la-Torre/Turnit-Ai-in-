"""Pruebas de integración con el modelo entrenado: necesitan GPU.

Correr con: pytest -m gpu
"""
import pytest

pytestmark = pytest.mark.gpu

HUMANO_LARGO = (
    "El empleo de modelos matemáticos para explicar fenómenos probabilísticos ha sido "
    "imprescindible en la investigación científica, aunque en el ámbito educativo es "
    "frecuente trabajar con variables que no cumplen los supuestos del modelo lineal. "
    "En este trabajo se revisan tres alternativas y se ilustra su uso con datos de una "
    "muestra de 214 estudiantes de secundaria de dos municipios del occidente del país. "
) * 6


@pytest.fixture(scope="module")
def detector():
    from detector.pipeline import Detector
    return Detector("cuda:1")


def test_texto_corto_no_emite_veredicto(detector):
    a = detector.analyze("Tres palabras aquí.")
    assert a.verdict == "insuficiente"
    assert a.warning


def test_texto_humano_no_se_marca(detector):
    a = detector.analyze(HUMANO_LARGO)
    assert a.verdict in ("humano", "gris")
    assert a.percent_ai < 50


def test_los_niveles_son_monotonos(detector):
    """Lo marcado solo puede crecer al aflojar el umbral."""
    normal = detector.analyze(HUMANO_LARGO, nivel="normal")
    estricto = detector.analyze(HUMANO_LARGO, nivel="estricto")
    exhaustivo = detector.analyze(HUMANO_LARGO, nivel="exhaustivo")
    assert normal.percent_ai <= estricto.percent_ai <= exhaustivo.percent_ai


def test_la_calibracion_trae_los_tres_niveles(detector):
    for nivel in ("1.0%", "5.0%", "10.0%"):
        assert nivel in detector.cal["thresholds"]
    # Umbral más exigente = valor más alto.
    t = detector.cal["thresholds"]
    assert t["1.0%"] > t["5.0%"] > t["10.0%"]


def test_el_analisis_devuelve_procedencia(detector):
    a = detector.analyze(HUMANO_LARGO)
    assert a.consistencia is not None
    assert a.consistencia.ventanas >= 1


def test_ventanas_cubren_el_texto(detector):
    a = detector.analyze(HUMANO_LARGO)
    assert a.windows
    assert sum(w.words for w in a.windows) > 0.8 * a.words
    for w in a.windows:
        assert 0.0 <= w.p_clf <= 1.0
        assert w.verdict in ("ia", "gris", "humano")
