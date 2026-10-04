"""Regla del veredicto de documento a partir de sus fragmentos.

Exige un mínimo de fragmentos marcados y no solo un porcentaje: en un texto corto un
único falso positivo por ventana ya sería el 25% del documento. Esa fue la corrección que
bajó los falsos positivos del 5,2% al 0,5% en textos humanos cortos.
"""
import pytest

from detector.pipeline import veredicto_por_fragmentos as regla


def test_todo_humano():
    assert regla(["humano"] * 8, [100] * 8) == "humano"


def test_un_solo_fragmento_marcado_en_texto_corto_no_basta():
    """Con 4 ventanas, una marcada es el 25% del texto: antes esto daba 'ia'."""
    assert regla(["ia", "humano", "humano", "humano"], [100] * 4) == "humano"


def test_dos_marcados_con_peso_suficiente_dan_zona_gris():
    """Dos de cuatro fragmentos es el 50% del texto: zona gris."""
    assert regla(["ia", "ia", "humano", "humano"], [100] * 4) == "gris"


def test_dos_marcados_sin_peso_no_escalan():
    """Dos de diez fragmentos es el 20%: por debajo del 30% que pide la regla."""
    assert regla(["ia", "ia"] + ["humano"] * 8, [100] * 10) == "humano"


def test_tres_marcados_con_peso_suficiente_dan_ia():
    assert regla(["ia"] * 3 + ["humano"] * 5, [100] * 8) == "ia"


def test_tres_fragmentos_cortos_en_documento_largo_no_escalan():
    """Tres fragmentos de 20 palabras en un documento de 3.000: el 2% del texto."""
    veredictos = ["ia"] * 3 + ["humano"] * 10
    palabras = [20] * 3 + [300] * 10
    assert regla(veredictos, palabras) == "humano"


def test_la_zona_gris_acumulada_cuenta():
    veredictos = ["ia", "ia", "gris", "gris", "humano"]
    assert regla(veredictos, [100] * 5) == "gris"


def test_documento_entero_marcado():
    assert regla(["ia"] * 6, [100] * 6) == "ia"


def test_sin_fragmentos_no_rompe():
    assert regla([], []) == "humano"


@pytest.mark.parametrize("n_ia,esperado", [(0, "humano"), (1, "humano"), (2, "gris"), (3, "ia")])
def test_umbral_de_fragmentos(n_ia, esperado):
    """Con 6 fragmentos iguales: 2 marcados son el 33% (gris) y 3 el 50% (ia)."""
    veredictos = ["ia"] * n_ia + ["humano"] * (6 - n_ia)
    assert regla(veredictos, [100] * 6) == esperado
