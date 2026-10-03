# Detector local de texto generado por IA

Herramienta local que marca los fragmentos de un texto que un detector de IA señalaría,
para reescribirlos antes de enviar el trabajo. **Uso previsto: revisión, no sanción.**
Prefiere marcar de más: un texto reformulado con una herramienta automática también se
marca, aunque las ideas sean propias.

**Informe completo del proyecto: [`docs/informe.md`](docs/informe.md)** — cómo decide,
el corpus, los resultados, los límites y cómo mantenerlo cuando salga un modelo nuevo.

## Estado

Detector funcionando, con la app local, el servidor MCP y el informe completo. Las cifras
de referencia (nivel normal, 1% de falsos positivos de diseño):

| | |
|---|---|
| IA sin parafrasear, ocho generadores | 99.5% detectada |
| Claude escribiendo desde cero (Opus 5, Sonnet 5, Haiku 4.5) | 99-100% |
| Texto humano pulido con IA | 80% (Claude) |
| IA parafraseada con DIPPER, dos pasadas | 73.9% |
| **Documentos reales escritos con IA en conversación** | **8%** (49% en nivel exhaustivo) |
| Falsos positivos: PERSUADE / tesis en español / CATyPI | 0.0 / 0.0 / 0.0% |
| Documentos largos humanos | 0 de 40 |
| Documentos con 30% de IA | 40 de 40 detectados |

Fases recorridas: Binoculars zero-shot → corpus pre-2022 con ocho generadores →
clasificador mDeBERTa y calibración → app local → textos de Claude → validación con
escritura de alumnos reales → ataques (frases, parafraseo, DIPPER) → entrenamiento
adversario → documentos reales del usuario → escritura fuertemente dirigida.

**El límite principal:** cuanto más dirige el autor la escritura, menos señal queda. Un
documento escrito al 100% con IA en conversación se marca al 8% en nivel normal. El
informe explica por qué no es un defecto corregible y cuál es la salida (preguntar
"¿escribe así este alumno?" en lugar de "¿lo escribió una máquina?").

**Señal de procedencia** (`detector/consistencia.py`): responde "¿una sola mano?" sin
datos de ningún alumno. Detecta el 93% de los documentos mixtos y marca el 7% de los
humanos. En los documentos reales del usuario es más informativa que el porcentaje
calibrado: el protocolo de tesis da 21 de 23 bloques con aspecto de máquina.

**Lo que queda pendiente:** trabajos de los propios alumnos del profesor para calibrar con
su población, y la verificación de autoría contra una muestra base por alumno.

## Qué incluye este repositorio, y qué no

Se publica **solo el código**: el detector, la app y todo el flujo de construcción y
evaluación del corpus.

No se publica, y hay que generarlo o conseguirlo por separado:

| | Dónde vive | Cómo obtenerlo |
|---|---|---|
| Modelo entrenado y `calibration.json` | `$DETECTOR_IA_HOME/models` | repositorios privados en Hugging Face (ver abajo), o reentrenar con `scripts/train_classifier.py` + `scripts/calibrate.py` |
| Corpus (artículos, tesis, textos de IA) | `$DETECTOR_IA_HOME/corpus` | `scripts/collect_openalex.py` y siguientes |
| Corpus externos (PERSUADE, CATyPI) | `$DETECTOR_IA_HOME/externo` | de sus autores; licencia CC BY-NC-SA, no redistribuibles |
| Trabajos analizados | nunca salen de la máquina | — |

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128  # según tu GPU
.venv/bin/pip install -r requirements.txt

export DETECTOR_IA_HOME=/ruta/con/espacio/detector-ia   # modelos y corpus (decenas de GB)
export OPENALEX_API_KEY=...                             # solo para construir el corpus
```

Hay **dos versiones** del modelo, ambas en repositorios privados de Hugging Face:

| | `ttech12/Ai-detector` (revisión) | `ttech12/Ai-detector-equilibrado` |
|---|---|---|
| IA parafraseada con DIPPER | **90.8%** | 5% |
| Texto humano reformulado con una herramienta | 40% IA + 30% gris | **0%** |
| Texto humano pulido con IA | 35% | 62% |
| Cuándo usarla | revisar un texto propio antes de enviarlo | cuando el resultado tenga consecuencias para alguien |

La de revisión aprendió a reconocer *que un texto pasó por un parafraseador*, no solo que
el contenido sea de máquina. Con acceso y sesión iniciada (`hf auth login`):

```bash
.venv/bin/hf download ttech12/Ai-detector \
  --local-dir "$DETECTOR_IA_HOME/models/mdeberta-detector"
mv "$DETECTOR_IA_HOME/models/mdeberta-detector/calibration.json" "$DETECTOR_IA_HOME/models/"
```

Se mantiene privado a propósito: si cualquiera puede descargarlo, también puede probar
textos contra él hasta encontrar cómo evadirlo.

## Uso

```bash
./run_app.sh      # app web local en http://127.0.0.1:8000


# Analizar un texto (puntuación más baja = más parecido a IA)
.venv/bin/python -m detector.cli trabajo.txt

# Rehacer la prueba de humo
.venv/bin/python scripts/fetch_smoke.py      # abstracts humanos pre-2022
.venv/bin/python scripts/generate_ai.py      # contraparte IA (requiere `ollama serve`)
.venv/bin/python -m scripts.eval_smoke       # AUROC y detección por idioma/generador
```

## Construir el corpus (fase 2)

Los datos viven en `$DETECTOR_IA_HOME/corpus`.
La clave de OpenAlex se lee de la variable `OPENALEX_API_KEY`.

```bash
.venv/bin/python -m scripts.collect_openalex          # resúmenes + lista de PDFs de tesis
.venv/bin/python -m scripts.fetch_theses --lang es    # fragmentos de tesis (y --lang en)
.venv/bin/python -m scripts.build_corpus              # dedup + particiones -> human.jsonl
.venv/bin/python -m scripts.generate_corpus           # contraparte IA -> ai.jsonl (reanudable)
.venv/bin/python -m scripts.finalize_corpus           # limpieza -> dataset.jsonl
.venv/bin/python -m scripts.eval_binoculars           # línea base en la partición test
```

`generate_corpus` necesita dos servidores Ollama:

```bash
# gpt-oss y qwen3 (modelos del sistema, solo lectura) en la GPU 1
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NOPRUNE=1 \
  OLLAMA_MODELS=/usr/share/ollama/.ollama/models CUDA_VISIBLE_DEVICES=1 ollama serve
# llama3.1, gemma3 y mistral-nemo en la GPU 0, puerto 11435
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_MODELS=/mnt/training_data/ollama-models CUDA_VISIBLE_DEVICES=0 ollama serve
```

Notas:
- Las particiones train/val/test se asignan por documento de origen (hash del id).
- `mistral-nemo:12b` solo aparece en test: mide la generalización a un modelo no visto.
- `ai_polished` = texto humano reescrito por IA; se evalúa aparte, es el caso ambiguo.
- `scripts/common.py` fuerza IPv4: el IPv6 de esta red hace que cada conexión tarde ~40 s.

## Cómo decide

Por cada ventana de oraciones se combinan el clasificador y Binoculars con los pesos de
`calibration.json`, y se compara contra dos umbrales fijados en validación: el del 1% de
falsos positivos (`ia`) y el del 5% (`gris`). El veredicto del documento toma lo más
severo entre la puntuación del texto completo y lo marcado por fragmentos (3 o más
fragmentos en `ia` y >=25% del texto), porque en un trabajo escrito a medias la
puntuación global se diluye. Se exige un mínimo de fragmentos, no solo un porcentaje:
en un texto corto un único falso positivo por ventana ya sería el 25% del documento.
Menos de 150 palabras: no se emite veredicto.

## Estructura

- `detector/pipeline.py`: análisis de un documento (ventanas, ensamble, veredictos).
- `detector/binoculars.py`: puntuación Binoculars global y por ventanas de oraciones.
- `detector/cli.py`: análisis de un archivo desde terminal.
- `app/`: app web local (FastAPI); `run_app.sh` la arranca.
- `scripts/`: descarga de datos, generación de textos IA y evaluación.
- `data/smoke/`: datos de la prueba de humo (`human.jsonl`, `ai.jsonl`, `scores_*.jsonl`).
- `scripts/eval_documents.py`: evaluación a nivel de documento completo.
- `scripts/eval_claude.py` + `PROMPT_AGENTE_CLAUDE.md`: corpus y evaluación con Claude.
- `scripts/import_claude.py`: integra los textos de Claude al corpus.
- `scripts/eval_alumnos.py`: falsos positivos sobre escritura de alumnos reales.
- `scripts/fetch_theses_eval.py`: baja tesis en español que no están en el corpus.
- `scripts/import_catypi.py`: extrae el corpus CATyPI (TEI XML) de escritura estudiantil.
- Corpus externo en `/mnt/training_data/detector-ia/externo` (PERSUADE 2.0, CC BY-NC-SA
  4.0: uso no comercial, no redistribuir).
- Modelo anterior (sin Claude) guardado en `models/mdeberta-detector-v1-sin-claude`.
