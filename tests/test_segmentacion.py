"""División en oraciones: los offsets tienen que poder reconstruir el texto."""
from detector.binoculars import split_sentences


def test_offsets_recuperan_las_oraciones():
    texto = "Primera oración. Segunda oración, con coma. ¿Y una pregunta? Sí."
    spans = split_sentences(texto)
    assert len(spans) >= 3
    for a, b in spans:
        assert texto[a:b].strip()
        assert 0 <= a < b <= len(texto)
    # Los tramos van en orden y no se solapan.
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b1 <= a2


def test_no_corta_en_abreviaturas_ni_decimales():
    texto = "El valor fue de 4.2 puntos en la métrica. Después subió."
    assert len(split_sentences(texto)) == 2


def test_texto_sin_puntuacion_da_una_sola_oracion():
    assert len(split_sentences("un texto sin punto final")) == 1


def test_apertura_en_espanol():
    texto = "Primera. ¿Segunda? ¡Tercera! Cuarta."
    assert len(split_sentences(texto)) == 4
