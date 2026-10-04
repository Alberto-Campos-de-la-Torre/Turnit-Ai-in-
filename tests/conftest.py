"""Fixtures y textos de muestra para las pruebas.

Las pruebas marcadas con `gpu` necesitan el modelo entrenado y una GPU libre; el resto
corre en segundos sin cargar nada. Para correr todas: pytest -m "" o pytest -m gpu
"""
import pytest

HUMANO = (
    "El empleo de modelos matemáticos para explicar fenómenos probabilísticos ha sido "
    "imprescindible en la investigación científica. No obstante, en el ámbito educativo es "
    "frecuente trabajar con variables que no cumplen las características requeridas por el "
    "modelo lineal clásico, utilizado durante mucho tiempo como única opción para "
    "representar datos de dependencia. En este trabajo se revisan tres alternativas y se "
    "ilustra su uso con datos de una muestra de 214 estudiantes de secundaria. "
)
IA = (
    "This study presents a comprehensive framework for evaluating neural architectures in "
    "low-resource settings. We propose a modular pipeline that combines transfer learning "
    "with parameter-efficient fine-tuning, and we evaluate it on four public benchmarks. "
    "Results demonstrate consistent improvements over strong baselines, with gains of up to "
    "4.2 points in macro F1. The findings highlight the importance of careful validation. "
)


@pytest.fixture
def texto_humano():
    return HUMANO * 4          # suficiente para pasar el mínimo de palabras


@pytest.fixture
def texto_ia():
    return IA * 4
