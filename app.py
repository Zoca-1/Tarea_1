"""Dashboard Streamlit · Clasificador de titulares peruanos (Tarea 1 PNL, Opción C).

Ejecutar:  streamlit run app.py
Requiere haber ejecutado 01_recoleccion.ipynb y 02_clasificacion.ipynb (genera modelo/ y datos/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src"))
import nlp_utils as nu  # noqa: E402

DATOS, MODELO = RAIZ / "datos", RAIZ / "modelo"
ETQ = nu.ETIQUETAS
COLOR_CAT = {"politica": "#2a78d6", "economia": "#eb6834", "deportes": "#1baf7a", "ciencia_tecnologia": "#eda100"}
COLOR_MEDIO = {"La República": "#2a78d6", "El Comercio": "#eb6834"}
COLOR_ENT = {"PER": "#2a78d6", "ORG": "#eb6834", "LOC": "#1baf7a"}
NOMBRE_ENT = {"PER": "Persona", "ORG": "Organización", "LOC": "Lugar"}

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="📰", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 3.5rem; max-width: 1250px;}
.titulo {text-align:center; font-size:2.1rem; font-weight:800; line-height:1.2; margin-bottom:.3rem;}
.subtitulo {text-align:center; color:#6b6b6b; font-size:1.02rem; margin-bottom:1.2rem;}
.badge {display:inline-block; padding:.25rem .7rem; border-radius:999px; color:white;
        font-weight:600; margin:.15rem .25rem .15rem 0; font-size:.9rem;}
.tarjeta {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:.8rem 1rem; margin-bottom:.6rem;}
.tarjeta small {color:#888;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Carga de datos y modelos (cacheados)
# --------------------------------------------------------------------------- #
@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATOS / "noticias.csv").fillna("")
    llm = pd.read_csv(DATOS / "etiquetas_llm.csv") if (DATOS / "etiquetas_llm.csv").exists() else None
    if llm is not None and len(llm) == len(df):
        df["etiqueta_llm"], df["confianza_llm"] = llm["etiqueta_llm"], llm["confianza"]
    if (DATOS / "particiones.csv").exists():
        df = df.merge(pd.read_csv(DATOS / "particiones.csv"), on="url", how="left")
    ents = pd.read_csv(DATOS / "entidades_ner.csv") if (DATOS / "entidades_ner.csv").exists() else None
    emb = np.load(DATOS / "embeddings.npy") if (DATOS / "embeddings.npy").exists() else None
    met = json.loads((MODELO / "metricas.json").read_text(encoding="utf-8")) \
        if (MODELO / "metricas.json").exists() else {}
    return df, ents, emb, met


@st.cache_resource(show_spinner="Cargando BETO fine-tuned…")
def cargar_clasificador():
    if (MODELO / "model.safetensors").exists() or (MODELO / "pytorch_model.bin").exists():
        return nu.ClasificadorTitulares(MODELO), "BETO fine-tuned"
    from transformers import pipeline  # respaldo si no se entrenó el modelo
    return pipeline("zero-shot-classification", model=nu.MODELO_ZERO_SHOT), "Zero-shot XNLI (respaldo)"


@st.cache_resource(show_spinner="Cargando modelo NER…")
def cargar_ner():
    return nu.cargar_ner()


@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def cargar_buscador(textos: tuple, _emb):
    return nu.BuscadorSemantico(list(textos), embeddings=_emb)


def predecir(texto: str) -> dict:
    clf, tipo = cargar_clasificador()
    if tipo.startswith("BETO"):
        return clf.predecir(texto)
    cand = {"política": "politica", "economía": "economia", "deportes": "deportes",
            "ciencia y tecnología": "ciencia_tecnologia"}
    r = clf(texto, candidate_labels=list(cand), hypothesis_template="Esta noticia trata sobre {}.")
    return {cand[l]: s for l, s in zip(r["labels"], r["scores"])}


def badge(texto: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{texto}</span>'


df, ents, emb, met = cargar_datos()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="titulo">¿Pueden convertir titulares peruanos en un clasificador de noticias?</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">400 titulares de <b>La República</b> y <b>El Comercio</b> (Opción C) · '
    'scraping RSS + HTML → etiqueta por sección editorial → zero-shot y LLM como líneas base → '
    '<b>BETO fine-tuned</b> · 4 categorías: política, economía, deportes, ciencia y tecnología</div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Filtros")
    medios = st.multiselect("Medio", sorted(df.medio.unique()), default=sorted(df.medio.unique()))
    cats = st.multiselect("Categoría", ETQ, default=ETQ, format_func=lambda e: nu.NOMBRES[e])
    texto_filtro = st.text_input("Contiene la palabra", placeholder="p. ej. Congreso")
    st.divider()
    st.caption("Datos descargados el 2026-10-12. Modelo: `dccuchile/bert-base-spanish-wwm-cased` "
               "afinado 3 épocas sobre 280 titulares.")

filtrado = df[df.medio.isin(medios) & df.etiqueta.isin(cats)]
if texto_filtro:
    filtrado = filtrado[filtrado.titular.str.contains(texto_filtro, case=False, regex=False)]

# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
k1, k2, k3, k4 = st.columns(4)
k1.metric("Noticias (filtro / total)", f"{len(filtrado)} / {len(df)}")
if met:
    k2.metric("F1 macro en test · BETO", f"{met['beto']['f1_macro_test']:.3f}",
              delta=f"{met['beto']['f1_macro_test'] - met['zero_shot']['f1_macro_40']:+.3f} vs. zero-shot")
    k3.metric("Acuerdo LLM–humano (auditoría 20)", f"{met['llm']['acuerdo_auditoria_20']:.0%}")
    k4.metric("Acuerdo LLM–editorial (400)", f"{met['llm']['accuracy_400'] * 100 + 1e-9:.1f}%")
else:
    k2.warning("Ejecuta 02_clasificacion.ipynb para ver métricas.")

tabs = st.tabs(["🧪 Probador en vivo", "📊 Explorar dataset", "🔎 Buscador semántico",
                "⚖️ Sesgo editorial", "📋 Resultados"])

# --------------------------------------------------------------------------- #
# 1. Probador en vivo
# --------------------------------------------------------------------------- #
with tabs[0]:
    _, tipo_modelo = cargar_clasificador()
    st.subheader("Clasifica un titular nuevo")
    st.caption(f"Modelo activo: **{tipo_modelo}**")
    ejemplos = ["Congreso aprueba en primera votación el retorno a la bicameralidad",
                "BCR mantiene su tasa de referencia y el dólar cierra a la baja",
                "Universitario vence a Sporting Cristal y queda a un paso del título",
                "Científicos peruanos usan inteligencia artificial para detectar huaicos"]
    ej = st.selectbox("…o elige un ejemplo", ["(escribir el mío)"] + ejemplos)
    titular = st.text_area("Titular", value="" if ej == "(escribir el mío)" else ej, height=80,
                           placeholder="Escribe aquí un titular de prensa peruano")
    if st.button("Predecir categoría", type="primary", disabled=not titular.strip()):
        with st.spinner("Clasificando…"):
            probs = predecir(titular.strip())
            ents_vivo = nu.extraer_entidades(cargar_ner(), titular.strip())
        mejor, conf = next(iter(probs.items()))
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Categoría predicha**")
            st.markdown(badge(nu.NOMBRES[mejor], COLOR_CAT[mejor]), unsafe_allow_html=True)
            st.metric("Confianza", f"{conf:.1%}")
            if conf < 0.6:
                st.warning("Confianza baja: el titular podría pertenecer a varias secciones.")
            st.markdown("**Entidades detectadas (NER)**")
            if ents_vivo:
                st.markdown(" ".join(badge(f"{e['entidad']} · {NOMBRE_ENT[e['tipo']]}", COLOR_ENT[e["tipo"]])
                                     for e in ents_vivo), unsafe_allow_html=True)
            else:
                st.caption("No se detectaron personas, organizaciones ni lugares.")
        with c2:
            st.markdown("**Probabilidad por categoría**")
            for e, p in probs.items():
                st.progress(float(p), text=f"{nu.NOMBRES[e]} · {p:.1%}")

# --------------------------------------------------------------------------- #
# 2. Explorar dataset
# --------------------------------------------------------------------------- #
with tabs[1]:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        conteo = filtrado.groupby(["etiqueta", "medio"]).size().rename("n").reset_index()
        fig = px.bar(conteo, x="etiqueta", y="n", color="medio", barmode="group", text="n",
                     category_orders={"etiqueta": ETQ}, color_discrete_map=COLOR_MEDIO,
                     labels={"etiqueta": "Categoría", "n": "Noticias", "medio": "Medio"},
                     title="Noticias por categoría y medio (filtro actual)")
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.12), height=420)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(filtrado.assign(longitud=filtrado.titular.str.len()), x="longitud", color="medio",
                           barmode="overlay", opacity=0.65, color_discrete_map=COLOR_MEDIO,
                           labels={"longitud": "Caracteres del titular", "medio": "Medio"},
                           title="Longitud de los titulares")
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.12), height=420,
                          yaxis_title="Titulares")
        st.plotly_chart(fig, width="stretch")

    cols = [c for c in ["titular", "bajada", "medio", "seccion_original", "etiqueta", "etiqueta_llm",
                        "confianza_llm", "fecha", "url"] if c in filtrado]
    st.dataframe(filtrado[cols], width="stretch", hide_index=True, height=380,
                 column_config={"url": st.column_config.LinkColumn("url", display_text="abrir"),
                                "confianza_llm": st.column_config.ProgressColumn(
                                    "confianza LLM", min_value=0, max_value=1, format="%.2f")})
    st.download_button("⬇️ Descargar CSV filtrado", filtrado[cols].to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# 3. Buscador semántico
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Encuentra los titulares más parecidos por significado")
    st.caption("Embeddings `paraphrase-multilingual-MiniLM-L12-v2` + similitud coseno sobre los 400 titulares. "
               "No necesita compartir palabras exactas.")
    consulta = st.text_input("Consulta", placeholder="p. ej. crisis entre el Ejecutivo y el Congreso")
    k = st.slider("Resultados", 3, 10, 3)
    if consulta.strip():
        buscador = cargar_buscador(tuple(df.titular), emb)
        for i, s in buscador.buscar(consulta, k=k):
            r = df.iloc[i]
            st.markdown(
                f'<div class="tarjeta">{badge(nu.NOMBRES[r.etiqueta], COLOR_CAT[r.etiqueta])}'
                f'<b>{r.titular}</b><br><small>{r.medio} · {r.fecha} · similitud coseno '
                f'<b>{s:.3f}</b> · <a href="{r.url}" target="_blank">ver nota</a></small></div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 4. Comparador de sesgo editorial
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("La República vs. El Comercio")
    cat_sel = st.selectbox("Categoría a comparar", ["todas"] + ETQ,
                           format_func=lambda e: "Todas" if e == "todas" else nu.NOMBRES[e])
    sub = df if cat_sel == "todas" else df[df.etiqueta == cat_sel]
    lr, ec = sub[sub.medio == "La República"], sub[sub.medio == "El Comercio"]

    c1, c2 = st.columns(2)
    for col, nombre, g in [(c1, "La República", lr), (c2, "El Comercio", ec)]:
        with col:
            st.markdown(f"#### {nombre}")
            a, b, c = st.columns(3)
            a.metric("Long. media", f"{g.titular.str.len().mean():.0f} car.")
            b.metric("Con «:»", f"{g.titular.str.contains(':').mean():.0%}")
            if "etiqueta_llm" in g:
                c.metric("Acuerdo LLM", f"{(g.etiqueta == g.etiqueta_llm).mean():.0%}")

    if "confianza_llm" in sub:
        fig = px.box(sub, x="etiqueta", y="confianza_llm", color="medio", points="all",
                     color_discrete_map=COLOR_MEDIO, category_orders={"etiqueta": ETQ},
                     labels={"confianza_llm": "Confianza del LLM", "etiqueta": "Categoría", "medio": "Medio"},
                     title="¿Qué tan 'obvios' son los titulares de cada medio para el LLM?")
        fig.update_traces(marker=dict(size=4, opacity=0.5))
        fig.update_layout(template="plotly_white", height=400, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

    top_lr, top_ec = nu.log_odds_distintivas(lr.titular.tolist(), ec.titular.tolist(), top=10)
    c1, c2 = st.columns(2)
    for col, nombre, t in [(c1, "La República", top_lr), (c2, "El Comercio", top_ec)]:
        with col:
            t = t.assign(z=t.z.abs())
            fig = px.bar(t, x="z", y="palabra", orientation="h", color_discrete_sequence=[COLOR_MEDIO[nombre]],
                         labels={"z": "z-score log-odds", "palabra": ""},
                         title=f"Palabras distintivas de {nombre}")
            fig.update_layout(template="plotly_white", height=380, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, width="stretch")

    if ents is not None:
        e_sub = ents if cat_sel == "todas" else ents[ents.etiqueta == cat_sel]
        top = (e_sub.groupby(["medio", "entidad", "tipo"]).size().rename("n").reset_index()
                    .sort_values("n", ascending=False).groupby("medio").head(8))
        fig = px.bar(top, x="n", y="entidad", color="tipo", facet_col="medio", orientation="h",
                     color_discrete_map=COLOR_ENT, labels={"n": "menciones", "entidad": "", "tipo": "Tipo"},
                     title="Entidades más mencionadas por medio (NER)")
        fig.update_yaxes(matches=None, showticklabels=True, categoryorder="total ascending")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(template="plotly_white", height=420, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# 5. Resultados
# --------------------------------------------------------------------------- #
with tabs[4]:
    if met:
        st.subheader("Comparación de enfoques")
        tabla = pd.DataFrame(met["tabla_comparativa"])
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in tabla.columns if tabla[c].dtype.kind == "f"})
        c1, c2 = st.columns([1, 1.2])
        with c1:
            if (MODELO / "matriz_confusion.png").exists():
                st.image(str(MODELO / "matriz_confusion.png"), caption="Matriz de confusión de BETO en test (60 titulares)")
        with c2:
            st.markdown(f"""
**Metodología**
1. **Recolección:** RSS → sitemaps → HTML de sección → rastreo, con pausas de 1.5 s y User-Agent de estudiante.
2. **Etiqueta:** sección editorial del medio mapeada a 4 categorías; 50 noticias por medio y categoría.
3. **Partición:** 70/15/15 estratificada por medio × categoría (semilla 42).
4. **Zero-shot:** BETO-XNLI → F1 macro **{met['zero_shot']['f1_macro_40']:.3f}** (40 titulares).
5. **LLM débil:** {met['llm']['modelo']} → acuerdo con la sección editorial **{met['llm']['accuracy_400'] * 100 + 1e-9:.1f}%**.
6. **BETO fine-tuned:** F1 macro en test **{met['beto']['f1_macro_test']:.3f}**, ~{met['beto']['latencia_ms']:.0f} ms por titular en CPU.
""")
    else:
        st.info("Ejecuta 02_clasificacion.ipynb para generar modelo/metricas.json.")
