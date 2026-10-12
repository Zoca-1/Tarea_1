"""Utilidades NLP compartidas por 02_clasificacion.ipynb y app.py."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

ETIQUETAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {
    "politica": "Política",
    "economia": "Economía",
    "deportes": "Deportes",
    "ciencia_tecnologia": "Ciencia y tecnología",
}
MODELO_ZERO_SHOT = "Recognai/bert-base-spanish-wwm-cased-xnli"
MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"
MODELO_NER = "mrm8488/bert-spanish-cased-finetuned-ner"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM_LOCAL = "Qwen/Qwen2.5-3B-Instruct"


# --------------------------------------------------------------------------- #
# Supervisión débil con LLM
# --------------------------------------------------------------------------- #
PROMPT_SISTEMA = (
    "Eres un editor de un diario peruano. Tu tarea es asignar cada titular de prensa "
    "a exactamente una sección del periódico."
)
PROMPT_TAREA = """Clasifica el titular de un diario peruano en UNA categoría:
1 = politica: Gobierno, Congreso, presidente, ministros, elecciones, partidos, Fiscalía, Poder Judicial y procesos contra autoridades o figuras públicas.
2 = economia: dinero y negocios: dólar, inflación, BCR, MEF, impuestos, AFP, precios, empresas, consumidores, empleo, comercio, gas, minería, inversión, startups, ofertas y ventas.
3 = deportes: fútbol, Liga 1, selección, clubes, dirigentes deportivos, jugadores, tenis, vóley y demás competencias.
4 = ciencia_tecnologia: descubrimientos científicos, arqueología, espacio, salud e investigación, clima y medio ambiente, tecnología, IA, celulares, apps, redes sociales y videojuegos.

Ejemplos:
"MEF eleva su proyección de crecimiento para el próximo año" -> 2
"Premier acude al Congreso para pedir el voto de confianza" -> 1
"Alianza Lima anuncia a su nuevo entrenador para la temporada" -> 3
"Astrónomos detectan un exoplaneta con posible atmósfera de agua" -> 4

Titular: "{titular}"
Responde solo con el número de la categoría."""


class EtiquetadorLLM:
    """Etiquetador zero-shot con un LLM generativo.

    backend:
      - "qwen_local": Qwen2.5-Instruct vía transformers (CPU/GPU, sin API key). La confianza
        es la probabilidad normalizada del token de respuesta ("1".."4").
      - "gemini": Google Gemini (requiere GEMINI_API_KEY). Confianza auto-reportada.
      - "ollama": servidor local de Ollama (http://localhost:11434). Confianza auto-reportada.
    """

    def __init__(self, backend: str = "auto", modelo: str | None = None):
        if backend == "auto":
            backend = "gemini" if os.getenv("GEMINI_API_KEY") else "qwen_local"
        self.backend = backend
        if backend == "qwen_local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.modelo = modelo or MODELO_LLM_LOCAL
            self.tok = AutoTokenizer.from_pretrained(self.modelo)
            self.tok.padding_side = "left"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
            self.llm = AutoModelForCausalLM.from_pretrained(self.modelo, dtype=dtype).to(self.device)
            self.llm.eval()
            self.ids_opciones = [self.tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 5)]
        elif backend == "gemini":
            from google import genai

            self.modelo = modelo or "gemini-2.5-flash"
            self.cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        elif backend == "ollama":
            self.modelo = modelo or "qwen2.5:3b"
        else:
            raise ValueError(f"backend desconocido: {backend}")

    def _mensajes(self, titular: str) -> list[dict]:
        return [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": PROMPT_TAREA.format(titular=titular.replace('"', "'"))},
        ]

    def clasificar(self, titulares: list[str], batch_size: int = 8) -> list[tuple[str, float]]:
        if self.backend == "qwen_local":
            return self._clasificar_qwen(titulares, batch_size)
        return [self._clasificar_api(t) for t in titulares]

    def _clasificar_qwen(self, titulares, batch_size):
        import torch

        salida = []
        for i in range(0, len(titulares), batch_size):
            prompts = [
                self.tok.apply_chat_template(self._mensajes(t), tokenize=False, add_generation_prompt=True)
                for t in titulares[i:i + batch_size]
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.llm(**enc).logits[:, -1, :]
            probs = torch.softmax(logits[:, self.ids_opciones].float(), dim=-1).cpu().numpy()
            salida += [(ETIQUETAS[int(p.argmax())], float(p.max())) for p in probs]
        return salida

    def _clasificar_api(self, titular: str) -> tuple[str, float]:
        instruccion = (PROMPT_TAREA.format(titular=titular)
                       + '\nDevuelve JSON: {"categoria": <1-4>, "confianza": <0-1>}')
        if self.backend == "gemini":
            r = self.cliente.models.generate_content(
                model=self.modelo, contents=PROMPT_SISTEMA + "\n\n" + instruccion,
                config={"response_mime_type": "application/json", "temperature": 0})
            texto = r.text
        else:
            import requests

            r = requests.post("http://localhost:11434/api/chat", timeout=120, json={
                "model": self.modelo, "stream": False, "format": "json", "options": {"temperature": 0},
                "messages": [{"role": "system", "content": PROMPT_SISTEMA},
                             {"role": "user", "content": instruccion}]})
            texto = r.json()["message"]["content"]
        try:
            d = json.loads(texto)
            idx = int(d["categoria"]) - 1
            return ETIQUETAS[idx], float(d.get("confianza", np.nan))
        except (ValueError, KeyError, IndexError, TypeError):
            m = re.search(r"[1-4]", texto)
            return (ETIQUETAS[int(m.group()) - 1] if m else "politica"), float("nan")


# --------------------------------------------------------------------------- #
# Clasificador fine-tuned (BETO)
# --------------------------------------------------------------------------- #
class ClasificadorTitulares:
    def __init__(self, ruta_modelo: str | Path, max_len: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ruta_modelo)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
        self.modelo.eval()
        self.max_len = max_len
        self.id2label = {int(k): v for k, v in self.modelo.config.id2label.items()}

    def predecir_proba(self, textos: list[str], batch_size: int = 32) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), batch_size):
            enc = self.tok(textos[i:i + batch_size], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.modelo(**enc).logits
            salida.append(self.torch.softmax(logits, dim=-1).numpy())
        return np.vstack(salida)

    def predecir(self, texto: str) -> dict[str, float]:
        p = self.predecir_proba([texto])[0]
        return dict(sorted(((self.id2label[i], float(v)) for i, v in enumerate(p)),
                           key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# NER y búsqueda semántica
# --------------------------------------------------------------------------- #
def cargar_ner():
    from transformers import pipeline

    return pipeline("ner", model=MODELO_NER, aggregation_strategy="max")


def extraer_entidades(ner, texto: str, umbral: float = 0.6) -> list[dict]:
    ents = []
    for e in ner(texto):
        palabra = e["word"].replace(" ##", "").replace("##", "").strip()
        if e["entity_group"] in {"PER", "ORG", "LOC"} and e["score"] >= umbral and len(palabra) > 1:
            ents.append({"entidad": palabra, "tipo": e["entity_group"], "score": float(e["score"])})
    return ents


class BuscadorSemantico:
    def __init__(self, textos: list[str], embeddings: np.ndarray | None = None, modelo=None):
        from sentence_transformers import SentenceTransformer

        self.modelo = modelo or SentenceTransformer(MODELO_EMBEDDINGS)
        self.textos = list(textos)
        self.emb = embeddings if embeddings is not None else self.codificar(self.textos)

    def codificar(self, textos: list[str]) -> np.ndarray:
        return self.modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32)

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[int, float]]:
        q = self.codificar([consulta])[0]
        sims = self.emb @ q  # coseno: los vectores están normalizados
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


# --------------------------------------------------------------------------- #
# Vocabulario distintivo entre medios (log-odds con prior de Dirichlet, Monroe et al. 2008)
# --------------------------------------------------------------------------- #
def _stopwords_es() -> set[str]:
    try:
        from spacy.lang.es.stop_words import STOP_WORDS
        base = set(STOP_WORDS)
    except ImportError:
        base = set()
    return base | {"vs", "video", "hoy", "así", "tras", "qué", "cómo", "cuál", "dónde", "según"}


def tokenizar(texto: str, stop: set[str]) -> list[str]:
    toks = re.findall(r"[a-záéíóúüñ0-9]+", texto.lower())
    return [t for t in toks if t not in stop and len(t) > 2 and not t.isdigit()]


def log_odds_distintivas(textos_a: list[str], textos_b: list[str], top: int = 12,
                         alpha0: float = 50.0):
    """Palabras más asociadas a A frente a B (z-score del log-odds con prior informativo)."""
    import pandas as pd
    from collections import Counter

    stop = _stopwords_es()
    ca = Counter(t for x in textos_a for t in tokenizar(x, stop))
    cb = Counter(t for x in textos_b for t in tokenizar(x, stop))
    total = ca + cb
    na, nb, n = sum(ca.values()), sum(cb.values()), sum(total.values())
    filas = []
    for w, f in total.items():
        if f < 3:
            continue
        aw = alpha0 * f / n
        ya, yb = ca[w], cb[w]
        delta = (np.log((ya + aw) / (na + alpha0 - ya - aw))
                 - np.log((yb + aw) / (nb + alpha0 - yb - aw)))
        var = 1 / (ya + aw) + 1 / (yb + aw)
        filas.append({"palabra": w, "z": delta / np.sqrt(var), "frec_a": ya, "frec_b": yb})
    df = pd.DataFrame(filas).sort_values("z")
    return df.tail(top)[::-1].reset_index(drop=True), df.head(top).reset_index(drop=True)
