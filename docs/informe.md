# Detector local de texto generado con IA — informe del proyecto

Herramienta local que marca los fragmentos de un texto que un detector de IA señalaría,
para un profesor sin acceso a Turnitin. Todo corre en una sola máquina: ningún trabajo de
alumno sale de ella.

**Uso previsto: revisión, no sanción.** El autor reescribe los fragmentos marcados hasta
que el indicador baja, y así el documento queda listo para enviarse a una revista sin que
lo señalen como escrito con IA. Esa decisión explica el sesgo del diseño: se prefiere
marcar de más. Un texto reformulado con una herramienta automática se marca aunque las
ideas sean del autor, porque las palabras no son suyas todavía.

**No sirve como prueba de deshonestidad, y no debe usarse así.** Que un texto salga limpio
no demuestra nada, y que salga marcado tampoco prueba nada por sí solo.

---

## 1. Qué hace y cómo decide

```
Documento (.docx / .pdf / texto)
        │
        ▼
Extracción de prosa ──► se descartan bibliografía, tablas, portada y encabezados
        │
        ▼
División en ventanas de oraciones (≥60 tokens)
        │
        ├─► Clasificador: mDeBERTa-v3-base afinado (humano vs IA)
        └─► Binoculars: zero-shot con dos LLM (Qwen2.5-3B base e instruct)
        │
        ▼
Ensamble logístico ──► umbral calibrado ──► veredicto por ventana y por documento
```

**Por ventana.** `puntuación = 2.596·logit(p_clasificador) − 2.726·binoculars − 19.510`
(pesos ajustados en validación), comparada con umbrales elegidos sobre textos humanos.

**Tres niveles de exigencia**, porque la escritura muy dirigida por el autor cae en zona
intermedia y el nivel más estricto no la recoge:

| Nivel | Falsos positivos de diseño | Umbral |
|---|---|---|
| `normal` | 1% | 1.14 |
| `estricto` | 5% | −3.49 |
| `exhaustivo` | 10% | −23.93 |

**Por documento.** El veredicto es lo más severo entre la puntuación del texto completo y
lo que marcan las ventanas: `ia` con 3 o más ventanas marcadas que sumen ≥25% del texto,
`gris` con 2 ventanas y ≥30%. Se exige un mínimo de ventanas y no solo un porcentaje,
porque en un texto corto (4 ventanas) un único falso positivo ya sería el 25% del
documento. Con menos de 150 palabras no se emite resultado.

**Por qué dos métodos.** El clasificador hace casi todo el trabajo. Binoculars no necesita
entrenamiento y sirve de red de seguridad ante un modelo de escritura desconocido, pero
tiene un punto ciego grave (sección 5).

---

## 2. El corpus

Todo el texto humano es **anterior a 2022**, antes de que existiera ChatGPT. En las tesis
se comprueba además la fecha de creación del PDF.

| | Humano | IA | IA "pulida" |
|---|---|---|---|
| Resúmenes en español | 3.112 | 3.389 | 727 |
| Fragmentos de tesis en español | 1.380 | 1.701 | 376 |
| Resúmenes en inglés | 4.500 | 5.575 | 1.000 |
| Fragmentos de tesis en inglés | 2.467 | 3.425 | 586 |
| **Total** | **11.459** | **14.090** | **2.689** |

- **Fuentes humanas:** OpenAlex (artículos y tesis, idioma verificado con lingua, menos de
  50 citas para evitar textos memorizados por los LLM) y el texto completo de tesis de
  acceso abierto, del que se extraen fragmentos de prosa.
- **Contraparte de IA:** por cada texto humano, un LLM escribe otro con el mismo título,
  idioma y extensión, de modo que el modelo aprenda el estilo y no el tema.
- **Ocho generadores:** gpt-oss:20b, qwen3:14b, qwen3:32b, gemma3:12b, llama3.1:8b,
  mistral-nemo:12b, **Claude** (1.800 textos escritos por subagentes) y **DIPPER**
  (1.381 textos de IA parafraseados, entrenamiento adversario).
- **`ai_polished`:** texto humano reescrito por la IA. Cuenta como IA en el entrenamiento;
  es la pieza que permite detectar fraseo de máquina sobre contenido humano (sección 6).
- **Particiones:** 70/15/15 por documento de origen, así que todos los fragmentos de una
  misma tesis caen en la misma partición.

### Conjuntos de evaluación externos

| Conjunto | Qué es | Tamaño |
|---|---|---|
| PERSUADE 2.0 | Ensayos de alumnos de 6.º a 12.º, EE.UU., inglés, 2010-2020 | 1.200 |
| Tesis en español ajenas al corpus | Licenciatura y maestría, no vistas en entrenamiento | 973 |
| CATyPI (INAOE) | Secciones de tesis de computación en español | 182 |
| **Documentos reales del usuario** | 11 trabajos propios escritos con IA en conversación | 22.297 palabras |
| **Prosa de transcripciones** | Prosa escrita por Claude en conversaciones reales | 17 bloques |

---

## 3. Qué detecta

Partición de prueba, nivel normal (1% de falsos positivos de diseño):

| | Detectado |
|---|---|
| IA sin parafrasear (todos los generadores) | 99.5% |
| Claude escribiendo desde cero (Opus 5, Sonnet 5, Haiku 4.5) | 99-100% |
| Mistral Nemo, generador no visto en entrenamiento | 99.6% |
| Texto humano pulido con IA | 80% (Claude), 45% (conjunto de prueba) |
| IA parafraseada con DIPPER, dos pasadas | 73.9% |
| **Documentos reales escritos con IA en conversación** | **8%** (49% en nivel exhaustivo) |

## 4. Falsos positivos

| Corpus | n | normal | estricto | exhaustivo |
|---|---|---|---|---|
| PERSUADE (ensayos escolares, inglés) | 300 | 0.0% | 0.3% | 2.0% |
| Tesis en español ajenas al corpus | 300 | 0.0% | 3.0% | **14.7%** |
| CATyPI (computación, español) | 263 | 0.0% | 1.1% | 4.6% |
| Documentos completos humanos (>1.200 palabras) | 40 | 0% | — | — |
| Documentos con 30% de IA (detectados) | 40 | 100% | — | — |

**Los falsos positivos se concentran en quienes escriben bien.** En PERSUADE la zona gris
aparece en el 2.6% de los ensayos con mejor nota y en el 0% de los peores. Si la
herramienta señala a alguien, es más probable que sea un alumno de los buenos.

**El nivel exhaustivo marca 1 de cada 7 fragmentos de tesis humanas auténticas.** Solo
tiene sentido para revisar un texto propio; usarlo para juzgar el trabajo de otro sería
indefendible.

---

## 5. Qué señales usa, y cuáles no

### El mito de las frases de longitud pareja

Circula la idea de que los detectores marcan un texto por tener frases regulares. La
regularidad **sí** distingue (variación de longitud: 0.406 en humanos, 0.244 en IA), pero
este detector apenas la usa: entre 1.633 textos humanos de prueba la correlación entre
variación y sospecha es −0.06, y manipular las frases de 300 textos de IA solo baja la
detección del 100% al 99.3%.

La documentación pública de Turnitin no describe perplejidad ni burstiness; eso procede de
blogs de posicionamiento. En el estudio comparativo revisado por pares más citado
(Weber-Wulff et al., 2023, *International Journal for Educational Integrity*) Turnitin fue
el mejor de 14 herramientas (76-81% de acierto) y el único que clasificó correctamente
todo el texto de IA editado a mano o parafraseado.

### El punto ciego de Binoculars

Binoculars mide predictibilidad. Sobre la prosa real escrita por Claude en conversación
con el autor da una mediana de **1.097**, es decir, **más humana que el texto humano
auténtico** (1.008). Cuando el autor decide el contenido, los datos y el fraseo turno a
turno, el texto se llena de decisiones que ningún modelo habría predicho, y la
predictibilidad desaparece. Ese límite aplica a cualquier detector de esta familia.

### Ataques probados

| Ataque | Resultado |
|---|---|
| Manipular la longitud de las frases | 100% → 99.3% |
| Parafrasear con un modelo conocido (gemma3:12b, dos pasadas) | 100% → 99.2% |
| **Parafrasear con DIPPER-XXL** (11B, hecho para evadir, no visto en entrenamiento) | 100% → **5%**; tras entrenamiento adversario, **73.9%** |

El contraste entre las dos últimas filas es la lección: **medir ataques con herramientas
que el modelo ya conoce da una falsa sensación de robustez.**

---

## 6. Los seis hallazgos que cambiaron el diseño

**1. El detector no generalizaba a Claude.** Entrenado con seis modelos locales detectaba
el 99.6% de ellos y el 39% de Claude (5% en tesis en inglés). Había aprendido el estilo de
esos generadores, no "escritura de IA". Añadir 1.500 textos de Claude lo subió al 100%.

**2. El texto pulido se detecta en proporción a cuánto cambió.** Sobre 428 textos pulidos,
la detección baja del 81% (reescrito casi entero) al 39% (retoque mínimo), con correlación
−0.32. El detector mide **cuánto del texto es de máquina**.

**3. Calibrar con escritura estudiantil sería peor.** Fijar el umbral sobre 2.355 textos de
alumnos triplicaría los falsos positivos en CATyPI (1.1% → 3.3%). Un artículo publicado se
parece **más** a un texto de IA que un trabajo de alumno, porque pasó por revisión y
corrección de estilo, así que es el que fija la cola alta y da el umbral estricto.

**4. La extracción del PDF diluía el resultado.** En un protocolo de tesis real, un tercio
de las palabras eran bibliografía, tablas, cifras y portada, y entraban al cálculo como si
fueran prosa. `detector/extraccion.py` aplica a los documentos la misma limpieza que se usó
para construir el corpus.

**5. Simular la escritura iterativa con modelos locales empeoró las cosas.** 592
conversaciones de cuatro turnos con Gemma, Mistral y Llama bajaron los documentos reales
del 9% al 5% marcado y el ataque DIPPER del 90.8% al 84.9%. Se le enseñó un estilo que no
era el objetivo. Archivado en `/mnt/almacen/modelo_v4_iterativo_fallido`.

**6. `ai_polished` como IA recuperó la escritura dirigida.** Los 2.689 textos con contenido
humano y palabras de máquina estaban excluidos del entrenamiento; incluirlos obliga al
modelo a mirar el fraseo y no el contenido:

| | Sin pulido | Con pulido como IA |
|---|---|---|
| Prosa real dirigida: p_clasificador | 0.0003 | **0.4257** |
| Prosa real dirigida: AUROC vs alumnos reales | ≈ azar | **0.939** |
| Claude puliendo texto humano | 33% | **80%** |
| Falsos positivos (CATyPI / tesis / PERSUADE) | 0 / 0.1 / 0.2% | 0 / 0.1 / 0.1% |
| Ataque DIPPER (dos pasadas) | 90.8% | 73.9% |

Recuperar la escritura dirigida cuesta 17 puntos de resistencia al parafraseo adversario.
Se eligió esta versión porque la escritura dirigida es lo que ocurre de verdad, mientras
DIPPER exige buscar e instalar un modelo de 45 GB.

---

## 7. El límite de fondo

Un protocolo de tesis de 6.105 palabras **escrito al 100% con IA** en conversación se marca
al 20% en nivel normal y al 88% en nivel exhaustivo. Cuatro de los once documentos reales
del usuario salen limpios en los tres niveles.

La razón no es un defecto corregible: **cuanto más dirige el autor, menos señal queda.** Si
las ideas, la estructura, los datos y los criterios son suyos y la máquina solo redacta,
entonces la proporción de autoría de máquina es genuinamente baja, y el detector la mide
bien. La etiqueta "escrito con IA" y la pregunta "¿cuánto de esto lo escribió una máquina?"
no son la misma cosa.

De ahí la dirección estratégica pendiente: cambiar la pregunta de **"¿lo escribió una
máquina?"** a **"¿escribe así este alumno?"**, comparando el trabajo contra una muestra
conocida de su propia escritura. Es robusto a que salgan modelos nuevos y no se evade
dirigiendo mejor la IA, a cambio de pedirle al profesor una muestra base por alumno.

---

## 8. Señal de procedencia: ¿una sola mano?

Pregunta distinta y complementaria: en vez de "¿lo escribió una máquina?", **"¿está este
documento escrito por una sola mano?"**. No necesita datos de ningún alumno, no guarda
nada y se calcula dentro del propio documento (`detector/consistencia.py`).

**Qué se mide.** El documento se agrupa en bloques de ~150 palabras y se cuenta cuántos
superan una vara moderada (p_clasificador > 0.5), cuántos quedan claramente por debajo
(< 0.1) y cuál es la racha más larga de bloques marcados consecutivos. Hay mezcla cuando
aparece una racha de 2 o más junto a 2 o más bloques claramente humanos. Se exige **racha**
y no bloques sueltos porque la mezcla real viene en bloques, mientras los falsos positivos
del clasificador son aislados.

| Tipo de documento (>1.200 palabras) | n | Marcado como mezcla | Bloques de máquina |
|---|---|---|---|
| Humano | 15 | **7%** | 5% |
| 30% de IA intercalada | 15 | **93%** | 53% |
| 60% de IA intercalada | 15 | 93% | 58% |
| Enteramente de IA | 15 | 0% (correcto: una sola mano) | 99% |

**El tamaño del bloque es crítico.** Con las ventanas pequeñas del análisis principal
(≥60 tokens) los falsos positivos suben del 7% al **53%**, porque el clasificador es más
ruidoso por fragmento y aparecen marcados aislados. De ahí el reagrupamiento a 150 palabras.

**Lo que no funcionó, medido y descartado:** estilometría clásica por ventana (palabras
función, n-gramas de caracteres, ritmo, puntuación) con agrupación en dos grupos o saltos
entre vecinas, contra una prueba de permutación. Seis variantes, AUROC entre 0.29 y 0.59,
es decir azar. La causa: un documento académico humano ya es heterogéneo porque la
introducción, la metodología y las conclusiones se escriben distinto, y esa variación
legítima tapa la de procedencia.

**Sobre los 11 documentos reales del usuario**, esta señal resulta más informativa que el
porcentaje calibrado: el protocolo de tesis da 21 de 23 bloques con aspecto de máquina, y
dos tareas dan 7 de 7 y 6 de 6, es decir "homogéneo, y parece de máquina". Cuatro
documentos siguen saliendo limpios.

**Es un indicio secundario, nunca un veredicto:** marca el 7% de los documentos humanos, y
las causas legítimas de heterogeneidad (citas largas, secciones de naturaleza distinta,
trabajo en equipo, redacción separada en el tiempo) no se distinguen de la mezcla con IA.

### Lo que queda diseñado y sin construir

Comparar contra una muestra conocida de la escritura del alumno sería más robusto, pero
exige datos suyos. Se puede hacer sin guardar texto (solo un vector de ~100 cifras del que
no se reconstruye nada), y aun así: una huella de estilo es un dato personal, el
consentimiento en una relación profesor-alumno nunca es del todo libre, y la muestra base
tendría que escribirse en clase y en un entorno controlado o la calibración queda al revés.
Se deja sin construir hasta que exista una política institucional de consentimiento y
borrado.

## 9. Límites conocidos

- **La escritura fuertemente dirigida se detecta mal** (8% en nivel normal). Es hoy el
  límite principal.
- **El texto pulido con IA se escapa una vez de cada cinco** con Claude, y más cuanto menos
  se cambió el original.
- **DIPPER reduce la detección al 73.9%** en inglés con dos pasadas.
- **Menos de 150 palabras: sin resultado.** No hay señal suficiente.
- **Ningún trabajo de los alumnos del profesor.** El 0% de falsos positivos en CATyPI es la
  mejor aproximación disponible.
- **Caduca.** Cada familia de modelos nueva obliga a repetir la medición.
- **Solo español e inglés**, y solo prosa académica.

---

## 10. Uso y mantenimiento

```bash
./run_app.sh                       # app local en http://127.0.0.1:8000
```

Cuando salga un modelo nuevo y se quiera comprobar si el detector lo reconoce:

1. `PROMPT_AGENTE_CLAUDE.md` + `data/corpus/claude_jobs.jsonl` → encargar los textos.
2. `scripts/eval_claude.py --texts <archivo>` → medir con el umbral ya calibrado.
3. Si la detección cae: generar textos de entrenamiento, `scripts/import_claude.py`,
   `finalize_corpus`, `train_classifier --incluir-pulido`, `calibrate`, y volver a medir.

Verificaciones que conviene repetir tras cualquier reentrenamiento:

| Script | Qué comprueba |
|---|---|
| `scripts/calibrate.py` | pesos y umbrales, con métricas en test |
| `scripts/eval_reales.py` | los 11 documentos reales escritos con IA |
| `scripts/eval_transcripciones.py` | prosa dirigida escrita en conversación |
| `scripts/eval_claude.py` | Claude de una sola pasada |
| `scripts/attack_dipper.py --score` | parafraseo adversario |
| `scripts/eval_documents.py` | documentos largos, humanos y mixtos |
| `scripts/eval_alumnos.py` | falsos positivos en escritura estudiantil real |
| `scripts/probe_shortcuts.py` | que el modelo no se apoye en artefactos del corpus |
| `scripts/eval_consistencia.py` | la señal de procedencia (mezcla vs una sola mano) |

---

## 11. Pruebas automatizadas

```bash
pytest            # 61 pruebas en ~3 s, sin cargar modelos
pytest -m gpu     # 6 de integración con el modelo entrenado
```

Las rápidas no tocan la GPU: la API se prueba con un detector falso y el MCP con la red
simulada. Cubren la división en oraciones y sus offsets, la extracción de prosa (incluido
el filtro de encabezados repetidos, que compara las líneas sin sus dígitos para que
"página 3" y "página 4" cuenten como la misma), la regla del veredicto de documento, la
señal de procedencia, la limpieza del corpus y las particiones.

Varias pruebas fijan correcciones concretas para que no vuelvan: que un único fragmento
marcado en un texto corto no escale a "ia", que "Claro que el modelo base funciona" no se
confunda con un preámbulo, y que "como asistente de ingeniería" no se tome por una negativa
del modelo.

La suite encontró un error el mismo día que se escribió: al añadir el campo de procedencia,
el aviso de "texto demasiado corto" quedó asignado por posición al campo equivocado, así
que la app devolvía el formulario sin explicación. Estaba en producción local y nadie lo
había notado.

## 12. Servidor MCP

Expone el análisis como herramientas para un asistente. **No carga el modelo**: habla por
HTTP con la app local, así la GPU se usa una sola vez aunque haya varios clientes.

| Herramienta | Qué hace |
|---|---|
| `analizar_texto` | texto directo; parámetros `fragmentos` y `nivel` |
| `analizar_archivo` | .pdf, .docx o .txt del disco |
| `estado` | comprueba que la app esté levantada |

Las instrucciones del servidor le dicen al asistente que esto es una señal de revisión y
que nunca presente la salida como evidencia de plagio, porque un asistente que recibe un
"100% marcado" tiende a redactar conclusiones más duras de lo que el dato permite.

---

## 13. Notas de operación

- **Modelos y datos** en `DETECTOR_IA_HOME`. Dos versiones publicadas en repositorios
  privados de Hugging Face: `ttech12/Ai-detector` (revisión) y
  `ttech12/Ai-detector-equilibrado` (sin entrenamiento adversario: no marca al autor que
  reformula su propio texto, pero DIPPER la evade). Archivados en `/mnt/almacen`:
  `modelo_v5_pre_pulido` y `modelo_v4_iterativo_fallido`.
- **mDeBERTa se carga con `dtype=torch.float32`**: el checkpoint viene en fp16 y con esos
  pesos AdamW produce NaN en el primer paso.
- **DIPPER tiene dos trampas:** en int8 devuelve texto vacío (T5 desborda; usar bf16 con
  `device_map="auto"`) y su repositorio trae un tokenizador roto de 104 piezas (hay que
  usar el de `google/t5-v1_1-xxl`).
- **`scripts/common.py` fuerza IPv4.** El IPv6 de esta red no enruta y cada conexión de
  `requests` tardaba ~40 s.
- **Clave de OpenAlex** en la variable `OPENALEX_API_KEY`.
- **Corpus externos** (PERSUADE 2.0, CATyPI) bajo CC BY-NC-SA 4.0: uso no comercial, no
  redistribuir. No se publican en este repositorio.
- **Los informes de los subagentes que escriben corpus no son fiables:** cuatro de siete
  afirmaron validaciones que no se cumplían. Verificar siempre el archivo con un script.
- **Dos subagentes se negaron a generar secciones académicas completas** con el método de
  cuatro pasos, por parecerse al flujo de un servicio de redacción por encargo. La objeción
  acertaba en un punto: el paso de insertar frases del autor "casi literales" existía para
  romper la huella estilométrica. Esa vía se abandonó.
- **La prosa de las transcripciones** se extrae de los archivos escritos con Write/Edit
  (.tex, .md), no de los mensajes del asistente: en Claude Code el texto del documento va
  en la herramienta que escribe el archivo, no en el mensaje.
