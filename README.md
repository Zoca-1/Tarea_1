# Clasificador de titulares peruanos · Tarea 1 PNL (Opción C)

> **¿Pueden convertir titulares peruanos en un clasificador de noticias?**
> Sí: con 280 titulares de entrenamiento, BETO afinado alcanza **F1 macro = 0.952** en test, frente a 0.665 del zero-shot.

| | |
|---|---|
| **Opción** | C – Comparada: 200 noticias de *La República* + 200 de *El Comercio* = **400** |
| **Categorías** | `politica` · `economia` · `deportes` · `ciencia_tecnologia` (50 por medio y categoría) |
| **Stack** | requests · feedparser · BeautifulSoup · Qwen2.5-3B (LLM local) · HF Transformers (BETO) · sentence-transformers · NER BETO-CoNLL · Plotly · Streamlit |

---

## 1. Pregunta del proyecto y objetivo

**Pregunta:** ¿se puede convertir un conjunto pequeño de titulares de prensa peruana en un clasificador supervisado y confiable de secciones temáticas?

**Objetivo:** recolectar titulares reales de dos medios con líneas editoriales distintas, etiquetarlos a partir de la sección editorial, comparar tres enfoques (zero-shot, supervisión débil con LLM y *fine-tuning* de un transformer en español) y desplegar el mejor en un dashboard interactivo. Además, analizar en qué se diferencian ambos medios (Opción C).

## 2. Medios y fechas de extracción

| Medio | Fuentes usadas (en orden de prioridad) | Noticias |
|---|---|---|
| **La República** (`larepublica.pe`) | RSS por sección (`/rss/<seccion>.xml`) → sitemaps de noticias (`/sitemap/<seccion>.xml`) → portadas de sección → rastreo de notas relacionadas de la misma sección | 200 |
| **El Comercio** (`elcomercio.pe`) | RSS por categoría (`/arcio/rss/category/<seccion>/`) → archivo de sección (`/archivo/<seccion>/`) | 200 |

- **Fecha de descarga:** 2026-10-12, con 188 requests y 0 errores.
- **Fechas de publicación:** el 81% es de septiembre u octubre de 2026 (69% solo de octubre). Las 49 notas anteriores a agosto de 2026 (la más antigua es de marzo de 2023) son todas de La República, sobre todo de ciencia/tecnología (28). Se obtuvieron por rastreo de notas relacionadas, porque su portada de sección solo muestra 12 notas.
- **Buenas prácticas:** User-Agent que identifica al estudiante, `time.sleep(1.5)` antes de cada request, máximo 1 reintento, respeto de `robots.txt` (La República prohíbe paginar con `/pag/`, por eso se rastrean notas relacionadas en lugar de páginas siguientes). Los respaldos crudos (RSS/sitemaps `.xml`, portadas `.html.gz` y `candidatos_crudos.json` con la bitácora de requests) están en `datos/raw/` (~5 MB).

## 3. Hallazgos analíticos del EDA

1. **La política domina el flujo, pero la mayoría de noticias queda fuera del alcance.** En el sitemap general de La República (las últimas 200 notas), el **72.5%** pertenece a secciones fuera de las 4 categorías (sociedad, mundo, espectáculos). Entre las válidas, la **política concentra el 45%** y economía y ciencia/tecnología apenas el 16% cada una. Por eso se muestreó con cuota fija (50 por medio y categoría).
2. **Cada medio tiene un enfoque y un estilo editorial propios.** El Comercio dedica el **36%** de su oferta visible a tecnología, y el **58%** de la de La República está en economía y política. La República titula más largo (101.5 vs. 97.4 caracteres) y "para el buscador": sus palabras más distintivas son *dólar, tipo de cambio, compra, venta, canal, fecha*. El Comercio titula para la coyuntura (*Congreso, facultades, Wall Street*) y repite fórmulas deportivas: el 34% de sus titulares de deportes son *"resumen y goles… | VIDEO"*.
3. **La misma sección no significa lo mismo en ambos medios.** El LLM coincide con la sección *Economía* en el 78% de los casos en La República, pero solo en el **60%** en El Comercio. Esa sección funciona como cajón de sastre: incluye notas sobre el presupuesto del Minsa, El Niño, conciertos de BTS, feriados y columnas de opinión.

## 4. Limitaciones técnicas

1. **Sesgo de titulares cortos y fronteras difusas.** Los titulares de columna (*"La IA agnóstica"*, *"Norte conectado"*, de menos de 20 caracteres) no tienen léxico del dominio. Los 3 errores de BETO en test están en la frontera política ↔ economía (*"Ejecutivo pedirá facultades para… régimen tributario"*): el modelo se guía por el actor político y no por el tema económico. Además, con un test de 60 titulares cada error vale 1.7 puntos, y en validación el F1 fue 0.867, así que la cifra de 0.95 tiene un margen amplio.
2. **Archivos pesados de PyTorch.** `modelo/model.safetensors` pesa **439 MB**, por encima del límite de 100 MB de GitHub. Por eso está en `.gitignore` (`modelo/*.safetensors`, `*.bin`, `checkpoint-*/`) y hay que regenerarlo ejecutando `02_clasificacion.ipynb`. En el repo solo quedan `config.json`, el tokenizer, `metricas.json` y la matriz de confusión. Si falta el modelo, la app usa el zero-shot como respaldo y lo indica.
3. **Etiquetas y referencia "humana".** La etiqueta es la sección editorial del medio, que es ruidosa. La auditoría de 20 casos compara el LLM con esa sección (un juicio editorial humano); el notebook deja `REVISION_MANUAL` para que un revisor corrija casos y se recalcule el acuerdo. El prompt del LLM se eligió comparando dos versiones en 120 titulares del propio corpus, así que su acuerdo es algo optimista.
4. **Cobertura del scraping.** Los sitemaps de La República solo cubren unos 2 días y sus portadas muestran 12 notas, mientras que El Comercio expone archivos de 100. Por eso las notas de La República abarcan un periodo más largo (hasta 2023) y la "oferta visible" de cada medio no es directamente comparable; las comparaciones se hacen en porcentajes dentro de cada medio.

## 5. Cómo ejecutarlo localmente

```bash
# 1) Entorno (Python 3.14 probado; 3.11+ debería funcionar)
cd tarea_1
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
# opcional, GPU NVIDIA:
# pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130

# 2) Regenerar el modelo (obligatorio la primera vez: los pesos no están en GitHub)
python scripts/ejecutar_notebooks.py 02     # ≈1 min en RTX 3060; en CPU tarda más
#   - 01_recoleccion.ipynb reutiliza datos/raw/ (REDESCARGAR=False). Pon REDESCARGAR=True para volver a scrapear.
#   - 02 reutiliza datos/etiquetas_llm.csv (REETIQUETAR_LLM=False). El LLM local necesita ~8 GB de VRAM.
#     Con GEMINI_API_KEY definida usa Gemini; también admite Ollama (backend="ollama").

# 3) Lanzar el dashboard
streamlit run app.py
```

La app abre en `http://localhost:8501` y tiene cinco pestañas: **Probador en vivo** (categoría, barra de confianza y entidades NER del titular), **Explorar dataset** (gráficos, tabla y descarga CSV), **Buscador semántico** (top-k por similitud coseno), **Sesgo editorial** (La República vs. El Comercio) y **Resultados**. Los filtros por medio y categoría están en la barra lateral.

## 6. Resumen de resultados

| Método | Evaluado en | Accuracy | **F1 macro** | Latencia/titular |
|---|---|---|---|---|
| Zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli` | 40 titulares de test | 0.675 | **0.665** | ~31 ms (GPU) |
| LLM débil `Qwen2.5-3B-Instruct` (local) | 400 / test | 0.843 / 0.817 | **0.843 / 0.817** | ~118 ms (GPU) |
| Embeddings multilingües + LogReg *(extra)* | test (60) | 0.950 | **0.949** | ~9 ms |
| **BETO fine-tuned** `dccuchile/bert-base-spanish-wwm-cased` | test (60) | **0.950** | **0.952** | ~35 ms (**CPU**) |

- **Acuerdo LLM–humano (auditoría de 20 casos aleatorios): 80%** (16/20). Los 3 errores claros del LLM fueron siglas institucionales (ONPE, JNJ) y Poder Judicial, todos con confianza ≥ 0.93, así que la confianza del LLM no sirve como filtro.
- **Elección para producción: BETO fine-tuned.** Tiene la mejor precisión, corre en CPU sin costo por consulta y no depende de una API. *Embeddings + LogReg* empata en la práctica (57/60 aciertos cada uno) y es la alternativa liviana. El LLM queda como herramienta de pre-etiquetado con auditoría humana.
- **Innovación:** (1) análisis de sesgo editorial con log-odds de Dirichlet y confianza del LLM por medio, (2) NER peruano por categoría (la política se narra con personas, el deporte con clubes y la economía con lugares) y (3) buscador semántico con embeddings multilingües.

## Estructura

```
tarea_1/
├── 01_recoleccion.ipynb      # scraping en capas, limpieza, mapeo de secciones, EDA (Plotly), exportación
├── 02_clasificacion.ipynb    # partición, zero-shot, LLM débil + auditoría, BETO, evaluación, innovación, tabla final
├── app.py                    # dashboard Streamlit
├── src/
│   ├── scraping.py           # Descargador (pausa + reintento + respaldo), parsers RSS/sitemap/HTML, muestreo
│   └── nlp_utils.py          # EtiquetadorLLM, ClasificadorTitulares, NER, BuscadorSemantico, log-odds
├── scripts/ejecutar_notebooks.py
├── datos/
│   ├── raw/                  # respaldos crudos de la descarga
│   ├── noticias.csv          # 400 filas: titular, bajada, medio, seccion_original, url, fecha, etiqueta
│   ├── etiquetas_llm.csv     # titular, etiqueta_llm, confianza, modelo_llm, fecha
│   ├── particiones.csv       # url → train/val/test
│   ├── entidades_ner.csv     # entidades por titular
│   └── embeddings.npy        # 400 × 384
└── modelo/                   # config + tokenizer + metricas.json + matriz_confusion.png (pesos en .gitignore)
```
