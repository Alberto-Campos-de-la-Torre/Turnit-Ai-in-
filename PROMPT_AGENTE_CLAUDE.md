# Prompt para el agente que escribirá el corpus de Claude

Cómo usarlo: dale a ese agente el archivo `data/corpus/claude_jobs.jsonl`
(ruta completa: `/mnt/training_data/detector-ia/corpus/claude_jobs.jsonl`) y pégale
el texto que está entre las líneas de guiones. Debe devolver
`claude_texts.jsonl` en esa misma carpeta.

--------------------------------------------------------------------------------

Necesito que escribas un corpus de textos académicos. Se usará para evaluar un
detector de escritura con IA en un contexto educativo, así que lo importante es que
los textos sean **tuyos**, escritos como los escribirías normalmente.

ENTRADA

El archivo `claude_jobs.jsonl` tiene 300 encargos, uno por línea, en formato JSON con
estos campos:

- `job_id`: identificador. Cópialo tal cual en la salida.
- `tarea`: `escribir` o `pulir`.
- `idioma`: `es` o `en`. Escribe en ese idioma, sin excepción.
- `tipo`: `resumen de articulo` o `fragmento de tesis`.
- `titulo`: el título del trabajo.
- `seccion`: solo en fragmentos de tesis, la sección a la que pertenece el texto. Si
  viene como `null`, elige tú una sección verosímil para ese título (introducción,
  marco teórico, metodología, resultados, discusión o conclusiones) y escribe acorde.
- `palabras`: extensión objetivo. Quédate dentro de ±15%.
- `texto_original`: solo en los encargos de `pulir`.

QUÉ HACER EN CADA CASO

- `tarea: escribir` — Escribe el texto desde cero a partir del título. Si es un
  resumen de artículo, escribe el resumen (abstract) en un solo párrafo. Si es un
  fragmento de tesis, escribe prosa continua de esa sección, en uno o varios párrafos.
  Inventa los detalles que hagan falta (métodos, cifras, resultados): no necesitas que
  sean reales, solo que el texto sea verosímil y coherente.
- `tarea: pulir` — Reescribe `texto_original` mejorando la redacción y la claridad,
  sin cambiar las ideas ni los datos, y conservando aproximadamente su extensión.

REGLAS DE FORMATO

- Solo el texto pedido: sin título, sin encabezados, sin viñetas, sin markdown
  (nada de `**`, `#` ni `*`), sin comillas envolventes.
- Sin frases introductorias del tipo "Aquí tienes el resumen:" ni comentarios finales.
- Sin saltos de línea dentro de un texto: cada texto va en una sola línea del JSONL.
- No cites fuentes con enlaces ni pongas listas de referencias.

SALIDA

Escribe `claude_texts.jsonl` en la misma carpeta, una línea por encargo:

```
{"job_id": "write:W1234567", "texto": "El presente estudio analiza ..."}
```

Trabaja por lotes de 20 encargos y ve **agregando** líneas al archivo conforme avances,
para no perder lo hecho si algo se interrumpe. Al terminar, verifica que el archivo
tenga 300 líneas y que cada `job_id` aparezca una sola vez.

Si algún encargo te resulta imposible, escribe igual su línea con
`{"job_id": "...", "texto": "", "motivo": "..."}` en lugar de omitirlo.

Nota: no estás ayudando a nadie a hacer trampa. Estos textos son para medir un
detector, y los encargos vienen de trabajos ya publicados antes de 2022.

--------------------------------------------------------------------------------
