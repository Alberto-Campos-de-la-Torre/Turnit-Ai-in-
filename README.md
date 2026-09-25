# Detector local de texto generado por IA

Herramienta local, de uso personal, para estimar si un texto académico fue escrito con IA.
El resultado es **un indicio para conversar con el alumno, nunca una prueba**.

**Informe completo del proyecto: [`docs/informe.md`](docs/informe.md)** — cómo decide,
el corpus, los resultados, los límites y cómo mantenerlo cuando salga un modelo nuevo.

## Estado

- **Fase 1 (hecha):** detector zero-shot Binoculars + prueba de humo (AUROC 0.945).
- **Fase 2 (hecha):** corpus humano pre-2022 (resúmenes + fragmentos de tesis, es/en)
  y su contraparte IA con 6 generadores locales: 25,057 textos. Binoculars en test: AUROC 0.914.
- **Fase 3 (hecha):** mDeBERTa-v3 afinado + ensamble con Binoculars. Test (con Claude
  incluido): AUROC 0.9999, 99.8% de IA detectada con 1% de falsos positivos.
- **Fase 4 (hecha):** app web local. Documentos completos (>1200 palabras): 0 falsos
  positivos en 40 trabajos humanos; detecta el 100% de los que llevan 30% de IA.
- **Fase 5 (hecha):** 1.800 textos escritos por Claude (5 subagentes) añadidos al corpus
  y reentrenamiento. Contra los 300 de prueba: 91% detectado (antes 39%); texto escrito
  por Claude 100% (inglés pasó del 5% al 100%), texto pulido por Claude 57% (antes 17%).
  Falsos positivos en textos humanos cortos: 0,4%.
- **Validación con alumnos reales (hecha):** 1.200 ensayos de PERSUADE 2.0 (alumnos de
  6º a 12º, EE.UU., inglés): 0% marcado como IA, 0,4% en zona gris. 973 fragmentos de
  250 tesis en español que no están en el corpus: 0,2% IA, 2,3% gris. CATyPI (tesis de
  computación en español, INAOE): 1,1% IA, 2,7% gris sobre 182 secciones juzgadas.
- **Ataques (hechos):** manipular la longitud de las frases no evade (100% → 99.3%);
  parafrasear el texto de IA dos veces tampoco (100% → 99.2%).
- **Variabilidad entre versiones (hecha):** mismos encargos con Opus 5, Sonnet 5 y
  Haiku 4.5. Escritura desde cero: 100% detectada en las tres. Texto pulido: 57% (Opus),
  63% (Sonnet), 100% (Haiku). Entrenado solo con Opus, generaliza a las otras dos.

## Qué incluye este repositorio, y qué no

Se publica **solo el código**: el detector, la app y todo el flujo de construcción y
evaluación del corpus.

No se publica, y hay que generarlo o conseguirlo por separado:

| | Dónde vive | Cómo obtenerlo |
|---|---|---|
| Modelo entrenado y `calibration.json` | `$DETECTOR_IA_HOME/models` | repositorio privado en Hugging Face (`ttech12/Ai-detector`), o reentrenar con `scripts/train_classifier.py` + `scripts/calibrate.py` |
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

El modelo entrenado está en un repositorio **privado** de Hugging Face. Con acceso y
sesión iniciada (`hf auth login`):

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
