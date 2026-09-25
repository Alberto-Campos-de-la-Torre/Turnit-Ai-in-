# Detector local de texto generado con IA — informe del proyecto

Herramienta local para estimar si un trabajo académico fue escrito con IA, pensada para
un profesor sin acceso a Turnitin. Todo corre en esta máquina: ningún trabajo de alumno
sale de aquí.

**El resultado es un indicio para conversar con el alumno, nunca una prueba.**

---

## 1. Qué hace y cómo decide

```
Trabajo (.docx / .pdf / texto)
        │
        ▼
Extracción del texto ─► división en ventanas de oraciones (≥60 tokens)
        │
        ├─► Clasificador: mDeBERTa-v3-base afinado (humano vs IA)
        └─► Binoculars: detector zero-shot con dos LLM (Qwen2.5-3B base e instruct)
        │
        ▼
Ensamble logístico ─► umbral calibrado ─► veredicto por ventana y por documento
```

**Por ventana.** Cada ventana recibe una puntuación del ensamble
(`1.362·logit(p_clasificador) − 0.919·binoculars − 8.772`, pesos ajustados en
validación) y se compara con dos umbrales elegidos sobre textos humanos:

| Veredicto | Umbral | Falsos positivos de diseño |
|---|---|---|
| `ia` | −4.77 | 1% |
| `gris` | −20.67 | 5% |
| `humano` | por debajo | — |

**Por documento.** El veredicto es lo más severo entre la puntuación del texto completo
y lo que marcan las ventanas: `ia` si hay **3 o más ventanas marcadas** y suman ≥25% del
texto; `gris` con 2 ventanas y ≥30% (o ≥40% sumando la zona gris). Se exige un mínimo de
ventanas, no solo un porcentaje, porque en un texto corto (4 ventanas) un único falso
positivo ya sería el 25% del documento. Con menos de 150 palabras no se emite veredicto.

**Por qué dos métodos.** El clasificador hace casi todo el trabajo; Binoculars aporta
poco al ensamble (AUROC 0.89 por su cuenta) pero no necesita entrenamiento, así que sirve
de red de seguridad si aparece un modelo de escritura muy distinto a los conocidos.

---

## 2. El corpus

Todo el texto humano es **anterior a 2022**, antes de que existiera ChatGPT. Para las
tesis se comprueba además la fecha de creación del PDF.

| | Humano | IA | IA "pulida" |
|---|---|---|---|
| Resúmenes en español | 3.112 | 3.389 | 727 |
| Fragmentos de tesis en español | 1.380 | 1.701 | 376 |
| Resúmenes en inglés | 4.500 | 4.826 | 1.000 |
| Fragmentos de tesis en inglés | 2.467 | 2.793 | 586 |
| **Total** | **11.459** | **12.709** | **2.689** |

- **Fuentes humanas:** OpenAlex (artículos y tesis, idioma verificado con lingua, menos
  de 50 citas para evitar textos memorizados por los LLM) y el texto completo de tesis de
  acceso abierto, del que se extraen fragmentos de prosa descartando referencias, tablas,
  índices y portadas.
- **Contraparte de IA:** por cada texto humano, un LLM escribe otro con el mismo título,
  idioma y extensión. Así el modelo aprende a distinguir el **estilo**, no el tema.
- **Generadores:** gpt-oss:20b, qwen3:14b, qwen3:32b, gemma3:12b, llama3.1:8b,
  mistral-nemo:12b (local, vía Ollama) y **Claude** (1.800 textos escritos por subagentes).
- **`ai_polished`:** texto humano reescrito por la IA ("mejora la redacción"). No entra al
  entrenamiento, solo a la evaluación: es el caso ambiguo del alumno que corrige su texto.
- **Particiones:** 70/15/15 asignadas por documento de origen, de modo que todos los
  fragmentos de una misma tesis caen en la misma partición.

---

## 3. Resultados

### Detección (partición de prueba, umbral del 1% de falsos positivos)

| Generador | Detectado |
|---|---|
| Los seis modelos locales | 99.4% – 100% |
| Claude Opus 5 (escritura desde cero) | 100% |
| Claude Sonnet 5 | 100% |
| Claude Haiku 4.5 | 100% |
| Mistral Nemo (nunca visto en entrenamiento) | 99.2% |
| Texto pulido con IA | 57% (Opus), 63% (Sonnet) |

### Falsos positivos sobre escritura humana real

| Corpus | n | `ia` | `gris` |
|---|---|---|---|
| PERSUADE 2.0 (ensayos escolares, inglés) | 1.200 | 0.0% | 0.4% |
| Tesis en español ajenas al corpus | 973 | 0.2% | 2.3% |
| CATyPI (tesis de computación en español) | 182 | 1.1% | 2.7% |
| Documentos completos (>1.200 palabras) | 40 | 0% | 0% |
| Documentos con 30% de IA (detectados) | 40 | 100% | — |

**Los falsos positivos se concentran en los buenos escritores.** En PERSUADE, la zona
gris aparece en el 2.6% de los ensayos con las mejores notas y en el 0% de los peores.
Los dos textos de CATyPI marcados son prosa formal y ordenada, sin nada anómalo. Si la
herramienta señala a alguien, es más probable que sea un alumno de los buenos.

---

## 4. Tres hallazgos que cambiaron el diseño

**El detector no generalizaba a Claude.** Entrenado solo con los seis modelos locales
detectaba el 99.6% de ellos y apenas el 39% de Claude (5% en tesis en inglés). Había
aprendido el estilo de esos generadores, no "escritura de IA". Se resolvió añadiendo
1.500 textos de Claude al entrenamiento; la detección subió al 100%. La lección es que
el detector va siempre un paso por detrás: cada familia de modelos nueva exige medir y,
si hace falta, reentrenar.

**El texto pulido se detecta en proporción a cuánto cambió.** Sobre 428 textos pulidos:

| Cuánto conserva del original | Detectado |
|---|---|
| Menos del 25% (reescrito entero) | 81% |
| 55–70% | 59% |
| Más del 70% (retoque mínimo) | 39% |

La correlación es −0.32. El detector no falla: mide **cuánto del texto es de máquina**.
Un retoque ligero casi no es texto de máquina, y marcarlo sería lo incorrecto.

**Calibrar con escritura estudiantil sería peor.** Fijar el umbral sobre los 2.355 textos
de alumnos daría −19.55 en lugar de −4.77 y triplicaría los falsos positivos en CATyPI
(1.1% → 3.3%). Un artículo publicado se parece **más** a un texto de IA que un trabajo de
alumno, porque pasó por revisión y corrección de estilo; por eso fija la cola alta y da
el umbral estricto que conviene.

---

## 5. Qué señales usa, y cuáles no

Circula la idea de que los detectores marcan un texto por tener frases de longitud
parecida y párrafos regulares (es lo que hacen las herramientas de tipo "perplejidad y
burstiness", como GPTZero, que muestran esos números en pantalla). Se verificó sobre la
partición de prueba:

**La regularidad sí es una característica real del texto de IA:**

| | Palabras por frase | Variación de longitud |
|---|---|---|
| Humano | 25.2 | 0.406 |
| IA | 27.2 | 0.244 |
| Humano pulido con IA | 23.3 | 0.353 |

**Pero este detector apenas la usa.** Entre los 1.633 textos humanos de prueba, la
correlación entre la variación de longitud y la sospecha del clasificador es de −0.06.
Marca al 1.5% de los que escriben con frases más parejas frente al 0.5% de los más
variados: una diferencia pequeña, no un sesgo sistemático.

**Y no se le engaña manipulando las frases.** Uniendo y partiendo frases en 300 textos de
IA, la variación subió de 0.252 a 0.290 y la detección solo bajó del 100% al 99.3%.

**El parafraseo tampoco evade.** Se reescribieron 240 textos de IA con un LLM (una y dos
pasadas), que es lo que hacen QuillBot y similares:

| Versión | Detectado | Conserva del original |
|---|---|---|
| Original | 100% | 1.00 |
| Una pasada de parafraseo | 100% | 0.40 |
| Dos pasadas | 99.2% | 0.45 |

Es el ataque que tumbó a los detectores de 2023 (Krishna et al., NeurIPS 2023: DetectGPT
cayó del 70.3% al 4.6% con DIPPER). Aquí no funciona, y la razón es simple: parafrasear
texto de IA con una IA deja texto de IA. Lo que baja la detección no es reescribir el
texto, sino que **el contenido sea humano**: cuando la IA reescribe un texto de un
alumno, la detección cae al 57-63%, y menos aún cuanto más conserva del original.

Advertencia sobre esta prueba: el parafraseador usado (gemma3:12b) es uno de los
generadores del entrenamiento, así que su estilo le resulta familiar al clasificador. La
prueba limpia sería DIPPER, el parafraseador de esa literatura, que no cabe en disco
(45 GB frente a 39 GB libres).

Sobre Turnitin en concreto: su documentación pública no describe el uso de esas métricas,
y el estudio comparativo revisado por pares más citado (Weber-Wulff et al., 2023,
*International Journal for Educational Integrity*) tampoco las atribuye a Turnitin;
solo menciona la "burstiness" como un valor que muestran otras herramientas. En ese
estudio Turnitin fue el mejor de 14 sistemas (76-81% de acierto) y el único que clasificó
correctamente todos los documentos de IA editados a mano o parafraseados. Buena parte de
lo que se lee en internet sobre "cómo funciona Turnitin" procede de blogs de
posicionamiento, no de fuentes primarias.

## 6. Límites conocidos

- **El texto pulido con IA se escapa la mitad de las veces.** Es el uso más común.
- **Ningún texto de alumnos reales usando Claude desde el chat.** Todo el corpus de IA lo
  escribieron agentes con instrucciones detalladas; un alumno escribe dos líneas de
  prompt y retoca el resultado.
- **Ningún trabajo de los alumnos del profesor.** El 1.1% de CATyPI es la mejor
  aproximación disponible, con un intervalo de confianza amplio (2 casos de 182).
- **Menos de 150 palabras: sin veredicto.** No hay señal suficiente.
- **Caducidad.** Cada familia de modelos nueva obliga a repetir la medición.

---

## 7. Uso y mantenimiento

```bash
~/detector-ia/run_app.sh          # app local en http://127.0.0.1:8000
```

Cuando salga un modelo nuevo y se quiera comprobar si el detector lo reconoce:

1. `PROMPT_AGENTE_CLAUDE.md` + `data/corpus/claude_jobs.jsonl` → encargar los textos.
2. `scripts/eval_claude.py --texts <archivo>` → medir con el umbral ya calibrado.
3. Si la detección cae, generar textos de entrenamiento (partición `train`),
   `scripts/import_claude.py`, `scripts/finalize_corpus.py`, `scripts/train_classifier.py`,
   `scripts/calibrate.py` y volver a medir.

Verificaciones que conviene repetir tras cualquier reentrenamiento:

| Script | Qué comprueba |
|---|---|
| `scripts/eval_binoculars.py` | línea base sin entrenamiento |
| `scripts/calibrate.py` | pesos y umbrales, medidos en test |
| `scripts/eval_documents.py` | documentos largos, mixtos y humanos |
| `scripts/eval_alumnos.py` | falsos positivos en escritura estudiantil real |
| `scripts/probe_shortcuts.py` | que el modelo no se apoye en artefactos del corpus |

---

## 8. Notas de operación

- **Modelos y datos** en `/mnt/training_data/detector-ia`; el modelo anterior a la
  incorporación de Claude está en `models/mdeberta-detector-v1-sin-claude`.
- **mDeBERTa se carga con `dtype=torch.float32`**: el checkpoint viene en fp16 y con esos
  pesos AdamW produce NaN en el primer paso.
- **`scripts/common.py` fuerza IPv4.** El IPv6 de esta red no enruta y cada conexión de
  `requests` tardaba ~40 s.
- **Clave de OpenAlex** en `~/.config/openalex.env` (variable `OPENALEX_API_KEY`).
- **Corpus externos** en `/mnt/training_data/detector-ia/externo`: PERSUADE 2.0 y CATyPI,
  ambos CC BY-NC-SA 4.0 (uso no comercial, no redistribuir).
- **Los informes de los subagentes que escriben corpus no son fiables:** cuatro de siete
  afirmaron validaciones que no se cumplían. Verificar siempre el archivo con un script.
