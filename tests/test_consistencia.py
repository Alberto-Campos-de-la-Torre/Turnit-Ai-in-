"""Señal de procedencia: ¿una sola mano o mezcla?

El criterio exige una racha de bloques marcados junto a bloques claramente humanos,
porque la mezcla real viene en bloques y los falsos positivos del clasificador son
fragmentos aislados.
"""
from detector.consistencia import MIN_VENTANAS, analizar


def test_documento_humano_homogeneo():
    c = analizar([0.001] * 10)
    assert not c.mezcla
    assert c.etiqueta == "parece de una sola mano"
    assert c.marcados == 0


def test_documento_enteramente_de_maquina_no_es_mezcla():
    c = analizar([0.999] * 10)
    assert not c.mezcla                     # una sola mano, aunque sea de máquina
    assert c.etiqueta == "homogéneo, y parece de máquina"
    assert c.racha == 10


def test_mezcla_en_bloques():
    c = analizar([0.001, 0.002, 0.99, 0.98, 0.97, 0.001, 0.003])
    assert c.mezcla
    assert c.racha == 3
    assert c.humanos == 4


def test_un_falso_positivo_aislado_no_es_mezcla():
    """El caso que obligó a exigir racha: una ventana marcada entre muchas humanas."""
    c = analizar([0.001] * 5 + [0.99] + [0.001] * 5)
    assert not c.mezcla
    assert c.racha == 1


def test_dos_falsos_positivos_separados_tampoco():
    c = analizar([0.001, 0.99, 0.001, 0.001, 0.99, 0.001, 0.002])
    assert not c.mezcla


def test_texto_corto_avisa_y_no_concluye():
    c = analizar([0.5] * (MIN_VENTANAS - 1))
    assert c.aviso and not c.mezcla
    assert c.etiqueta == "sin datos suficientes"


def test_lista_vacia_no_rompe():
    c = analizar([])
    assert c.aviso and c.ventanas == 0
