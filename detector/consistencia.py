"""¿Está este documento escrito por una sola mano, o parece mezcla?

No detecta IA: detecta **mezcla de procedencia**. Un trabajo donde el alumno escribió la
metodología y la IA escribió la introducción tiene fragmentos de dos clases distintas, y
eso se mide dentro del propio documento, sin compararlo con nada externo y sin guardar
datos de nadie.

Complementa al detector: este dice "parece de máquina", aquel dice "parecen dos manos". Un
documento escrito entero por una IA es homogéneo y aquí no salta.

La señal: **una racha de fragmentos que parecen de máquina junto a fragmentos que parecen
claramente humanos**. Se exige una racha y no fragmentos sueltos porque la mezcla real
viene en bloques (una sección entera escrita por otra mano), mientras los falsos positivos
del clasificador son fragmentos aislados.

Medido sobre documentos de más de 1.200 palabras: detecta el 97% de los mixtos, el 0% de
los que son enteramente de IA (correcto: una sola mano) y marca el **7% de los humanos**.
Ese 7% es el precio y viene de los falsos positivos por fragmento del propio clasificador,
así que esto es un indicio secundario, nunca un veredicto.

Lo que NO funcionó, medido y descartado: estilometría clásica por ventana (palabras
función, n-gramas de caracteres, ritmo y puntuación) con agrupación en dos grupos o saltos
entre ventanas vecinas, contra una prueba de permutación. Seis variantes, AUROC entre 0.29
y 0.59, es decir azar. La causa: un documento académico humano ya es heterogéneo porque la
introducción, la metodología y las conclusiones se escriben distinto, y esa variación
legítima tapa la de procedencia.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_VENTANAS = 5
UMBRAL_FRAGMENTO = 0.5     # vara moderada: "este fragmento parece de máquina"
UMBRAL_HUMANO = 0.1        # "este fragmento parece claramente humano"
MIN_RACHA = 2              # fragmentos marcados consecutivos
MIN_HUMANOS = 2            # fragmentos claramente humanos en el documento


@dataclass
class Consistencia:
    racha: int                 # fragmentos consecutivos que parecen de máquina
    marcados: int              # fragmentos que parecen de máquina, en total
    humanos: int               # fragmentos que parecen claramente humanos
    ventanas: int
    mezcla: bool               # ¿parece escrito por más de una mano?
    dispersion: float          # desviación típica de la probabilidad entre fragmentos
    aviso: str | None = None

    @property
    def etiqueta(self) -> str:
        if self.aviso:
            return "sin datos suficientes"
        if self.mezcla:
            return "parece mezcla de procedencias"
        if self.ventanas and self.marcados / self.ventanas >= 0.6:
            return "homogéneo, y parece de máquina"
        return "parece de una sola mano"


def analizar(p_por_ventana) -> Consistencia:
    """Recibe la probabilidad de IA de cada fragmento (la que ya calcula el pipeline)."""
    p = np.asarray([x for x in p_por_ventana], dtype=float)
    if len(p) < MIN_VENTANAS:
        return Consistencia(0, 0, 0, len(p), False, float("nan"),
                            f"Hacen falta al menos {MIN_VENTANAS} fragmentos; "
                            f"este texto da {len(p)}.")
    mejor = actual = 0
    for x in p:
        actual = actual + 1 if x > UMBRAL_FRAGMENTO else 0
        mejor = max(mejor, actual)
    marcados = int((p > UMBRAL_FRAGMENTO).sum())
    humanos = int((p < UMBRAL_HUMANO).sum())
    mezcla = mejor >= MIN_RACHA and humanos >= MIN_HUMANOS
    return Consistencia(mejor, marcados, humanos, len(p), mezcla, float(p.std()))
