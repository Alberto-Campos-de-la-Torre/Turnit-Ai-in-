"""Limpieza del corpus: preámbulos, negativas del modelo y particiones."""
from scripts.build_corpus import split_of
from scripts.common import text_hash
from scripts.finalize_corpus import REFUSAL, clean_ai


def test_quita_preambulos():
    assert clean_ai("Aquí tienes el resumen: El presente trabajo") == "El presente trabajo"
    assert clean_ai("Here's a draft, about 300 words: The study") == "The study"


def test_no_corta_una_oracion_legitima_que_empieza_por_claro():
    """El filtro de preámbulos era demasiado amplio y se comía texto válido."""
    texto = "Claro que el modelo base funciona: no conviene tocarlo"
    assert clean_ai(texto) == texto


def test_quita_marcado_markdown():
    assert clean_ai("El **presente** trabajo con *énfasis*") == "El presente trabajo con énfasis"


def test_detecta_negativas_del_modelo():
    assert REFUSAL.search("Lo siento, pero no puedo generar ese contenido")
    assert REFUSAL.search("I cannot help with that request")
    assert REFUSAL.search("Unfortunately, I cannot write this")


def test_no_confunde_prosa_con_negativa():
    """"como asistente de ingeniería" no es una negativa: fue un falso positivo real."""
    assert not REFUSAL.search("El trabajo describe la práctica como asistente de ingeniería")
    assert not REFUSAL.search("No puedo evitar señalar que los datos son escasos")


def test_particiones_deterministas():
    assert split_of("W123456") == split_of("W123456")
    assert split_of("W123456#p0") in ("train", "val", "test")


def test_las_particiones_reparten_aproximadamente_70_15_15():
    from collections import Counter
    c = Counter(split_of(f"W{i}") for i in range(4000))
    assert 0.65 < c["train"] / 4000 < 0.75
    assert 0.12 < c["val"] / 4000 < 0.18
    assert 0.12 < c["test"] / 4000 < 0.18


def test_el_hash_ignora_puntuacion_y_mayusculas():
    assert text_hash("El presente trabajo.") == text_hash("el presente trabajo")
    assert text_hash("Un texto") != text_hash("Otro texto")
