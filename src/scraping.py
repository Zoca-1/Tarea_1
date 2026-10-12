"""Recolección de titulares de La República y El Comercio (Opción C).

Estrategia en capas, de la fuente más "oficial" a la más costosa:
  1. Feeds RSS por sección (feedparser).
  2. Sitemaps de noticias por sección (solo La República; permitidos en su robots.txt).
  3. Páginas HTML de sección / archivo (BeautifulSoup).
  4. Rastreo ligero de artículos de la misma sección (enlaces relacionados) cuando
     una categoría no llega a la cuota. También sirve para completar bajada y fecha.

Buenas prácticas: User-Agent de estudiante, pausa de 1.5 s antes de cada request,
máximo 1 reintento, y respaldo crudo de cada respuesta en datos/raw/.
"""
from __future__ import annotations

import gzip
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (compatible; TareaPNL-Estudiante/1.0; "
    "proyecto academico de clasificacion de titulares)"
)
PAUSA_SEG = 1.5
MAX_REINTENTOS = 1

LR = "La República"
EC = "El Comercio"

FUENTES = {
    LR: {
        "base": "https://larepublica.pe",
        "rss": {
            "politica": "https://larepublica.pe/rss/politica.xml",
            "economia": "https://larepublica.pe/rss/economia.xml",
            "deportes": "https://larepublica.pe/rss/deportes.xml",
            "ciencia": "https://larepublica.pe/rss/ciencia.xml",
        },
        "sitemaps": {
            "politica": "https://larepublica.pe/sitemap/politica.xml",
            "economia": "https://larepublica.pe/sitemap/economia.xml",
            "deportes": "https://larepublica.pe/sitemap/deportes.xml",
            "ciencia": "https://larepublica.pe/sitemap/ciencia.xml",
            "noticias": "https://larepublica.pe/sitemap/noticias.xml",
        },
        "html": {
            "politica": "https://larepublica.pe/politica",
            "economia": "https://larepublica.pe/economia",
            "economia/precio-del-dolar": "https://larepublica.pe/economia/precio-del-dolar",
            "deportes": "https://larepublica.pe/deportes",
            "ciencia": "https://larepublica.pe/ciencia",
            "tecnologia": "https://larepublica.pe/tecnologia",
        },
    },
    EC: {
        "base": "https://elcomercio.pe",
        "rss": {
            "politica": "https://elcomercio.pe/arcio/rss/category/politica/",
            "economia": "https://elcomercio.pe/arcio/rss/category/economia/",
            "deporte-total": "https://elcomercio.pe/arcio/rss/category/deporte-total/",
            "tecnologia": "https://elcomercio.pe/arcio/rss/category/tecnologia/",
        },
        "sitemaps": {},
        "html": {
            "politica": "https://elcomercio.pe/archivo/politica/",
            "economia": "https://elcomercio.pe/archivo/economia/",
            "deporte-total": "https://elcomercio.pe/archivo/deporte-total/",
            "tecnologia": "https://elcomercio.pe/archivo/tecnologia/",
            "tecnologia/ciencias": "https://elcomercio.pe/archivo/tecnologia/ciencias/",
        },
    },
}

# Equivalencias sección del medio (primer segmento de la URL) -> etiqueta del proyecto.
MAPEO_SECCIONES = {
    (LR, "politica"): "politica",
    (LR, "economia"): "economia",
    (LR, "deportes"): "deportes",
    (LR, "ciencia"): "ciencia_tecnologia",
    (LR, "tecnologia"): "ciencia_tecnologia",
    (EC, "politica"): "politica",
    (EC, "economia"): "economia",
    (EC, "deporte-total"): "deportes",
    (EC, "tecnologia"): "ciencia_tecnologia",
}
ETIQUETAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]

_RE_FECHA_URL = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def normalizar_url(url: str) -> str:
    p = urlparse(url.strip())
    path = p.path.rstrip("/")
    if "elcomercio.pe" in p.netloc:
        path += "/"  # El Comercio usa barra final canónica
    return f"https://{p.netloc}{path}"


def seccion_desde_url(url: str) -> str:
    """'/economia/mercados/slug-noticia/' -> 'economia/mercados'; '/economia/2026/10/..' -> 'economia'."""
    segs = [s for s in urlparse(url).path.split("/") if s]
    out = []
    for s in segs[:-1]:
        if re.fullmatch(r"\d{4}", s):
            break
        out.append(s)
    return "/".join(out[:2]) if out else (segs[0] if segs else "")


def mapear_etiqueta(medio: str, seccion_original: str) -> str | None:
    return MAPEO_SECCIONES.get((medio, seccion_original.split("/")[0]))


def es_articulo(medio: str, url: str) -> bool:
    path = urlparse(url).path
    if medio == LR:
        return bool(_RE_FECHA_URL.search(path)) and bool(re.search(r"-\d{4,}/?$", path))
    return path.rstrip("/").endswith("-noticia")


def limpiar_texto(txt: str | None) -> str:
    if not txt:
        return ""
    if "<" in txt:
        txt = BeautifulSoup(txt, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", txt).strip()


def fecha_iso(valor) -> str:
    if not valor:
        return ""
    if isinstance(valor, time.struct_time):
        return time.strftime("%Y-%m-%d", valor)
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(valor))
    return m.group(1) if m else ""


def fecha_desde_url(url: str) -> str:
    m = _RE_FECHA_URL.search(urlparse(url).path)
    return "-".join(m.groups()) if m else ""


def _slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Descargador con pausa, reintento y respaldo crudo
# --------------------------------------------------------------------------- #
class Descargador:
    def __init__(self, raw_dir: Path, pausa: float = PAUSA_SEG, reintentos: int = MAX_REINTENTOS):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pausa = pausa
        self.reintentos = reintentos
        self.sesion = requests.Session()
        self.sesion.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es-PE,es;q=0.9"})
        self.bitacora: list[dict] = []

    def get(self, url: str, nombre_raw: str | None = None) -> str | None:
        for intento in range(self.reintentos + 1):
            time.sleep(self.pausa)
            try:
                r = self.sesion.get(url, timeout=25)
                estado = r.status_code
                if estado == 200:
                    texto = r.content.decode("utf-8", errors="replace")
                    self.bitacora.append({"url": url, "estado": estado, "intento": intento})
                    if nombre_raw:
                        self._respaldar(nombre_raw, texto)
                    return texto
                if estado not in (429, 500, 502, 503, 504):
                    break  # 404/403: no tiene sentido reintentar
            except requests.RequestException as e:
                estado = f"error: {type(e).__name__}"
        self.bitacora.append({"url": url, "estado": estado, "intento": intento})
        return None

    def _respaldar(self, nombre: str, texto: str) -> None:
        destino = self.raw_dir / nombre
        if nombre.endswith(".gz"):
            with gzip.open(destino, "wt", encoding="utf-8") as f:
                f.write(texto)
        else:
            destino.write_text(texto, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Parsers por tipo de fuente
# --------------------------------------------------------------------------- #
def parsear_rss(texto: str, medio: str, fuente_seccion: str) -> list[dict]:
    feed = feedparser.parse(texto)
    registros = []
    for e in feed.entries:
        url = e.get("link", "")
        if not url or not es_articulo(medio, url):
            continue
        registros.append({
            "titular": limpiar_texto(e.get("title")),
            "bajada": limpiar_texto(e.get("summary") or e.get("description")),
            "medio": medio,
            "url": normalizar_url(url),
            "fecha": fecha_iso(e.get("published_parsed")) or fecha_desde_url(url),
            "fuente": f"rss:{fuente_seccion}",
        })
    return registros


def parsear_sitemap_noticias(texto: str, medio: str, fuente_seccion: str) -> list[dict]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
          "news": "http://www.google.com/schemas/sitemap-news/0.9"}
    raiz = ET.fromstring(texto.strip().encode("utf-8"))
    registros = []
    for u in raiz.findall("sm:url", ns):
        loc = (u.findtext("sm:loc", default="", namespaces=ns) or "").strip()
        if not loc or not es_articulo(medio, loc):
            continue
        titulo = u.findtext("news:news/news:title", default="", namespaces=ns)
        fecha = u.findtext("news:news/news:publication_date", default="", namespaces=ns)
        registros.append({
            "titular": limpiar_texto(titulo),
            "bajada": "",
            "medio": medio,
            "url": normalizar_url(loc),
            "fecha": fecha_iso(fecha.strip()) or fecha_desde_url(loc),
            "fuente": f"sitemap:{fuente_seccion}",
        })
    return registros


def parsear_html_seccion(texto: str, medio: str, fuente_seccion: str) -> list[dict]:
    base = FUENTES[medio]["base"]
    soup = BeautifulSoup(texto, "html.parser")
    registros = []
    if medio == EC:
        # Archivo de El Comercio: cada nota trae título, bajada y fecha.
        for a in soup.select("a.story-item__title"):
            url = urljoin(base, a.get("href", ""))
            if not es_articulo(medio, url):
                continue
            caja = a.find_parent("div", class_="story-item__left") or a.parent
            sub = caja.select_one(".story-item__subtitle") if caja else None
            t = caja.select_one("time[datetime]") if caja else None
            registros.append({
                "titular": limpiar_texto(a.get_text()),
                "bajada": limpiar_texto(sub.get_text()) if sub else "",
                "medio": medio,
                "url": normalizar_url(url),
                "fecha": fecha_iso(t["datetime"]) if t else "",
                "fuente": f"html:{fuente_seccion}",
            })
    else:
        for a in soup.find_all("a", href=True):
            url = urljoin(base, a["href"])
            if urlparse(url).netloc != "larepublica.pe" or not es_articulo(medio, url):
                continue
            texto_a = limpiar_texto(a.get_text())
            registros.append({
                # anclas cortas ("RJ Mitte") son etiquetas de tendencia, no titulares
                "titular": texto_a if len(texto_a.split()) >= 5 else "",
                "bajada": "",
                "medio": medio,
                "url": normalizar_url(url),
                "fecha": fecha_desde_url(url),
                "fuente": f"html:{fuente_seccion}",
            })
    return registros


def _meta(soup: BeautifulSoup, *claves: str) -> str:
    for k in claves:
        tag = soup.find("meta", attrs={"property": k}) or soup.find("meta", attrs={"name": k})
        if tag and tag.get("content"):
            return tag["content"]
    return ""


def parsear_articulo(texto: str, medio: str, url: str) -> tuple[dict, list[str]]:
    """Devuelve (metadatos del artículo, enlaces a otros artículos del mismo medio)."""
    soup = BeautifulSoup(texto, "html.parser")
    fecha = fecha_iso(_meta(soup, "article:published_time", "date")) or fecha_desde_url(url)
    if not fecha:
        for s in soup.find_all("script", type="application/ld+json"):
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', s.string or "")
            if m:
                fecha = fecha_iso(m.group(1))
                break
    titulo = _meta(soup, "og:title") or (soup.title.get_text() if soup.title else "")
    titulo = re.sub(r"\s*[|\-–]\s*(La República|El Comercio( Perú)?)\s*$", "", titulo)
    meta = {
        "titular": limpiar_texto(titulo),
        "bajada": limpiar_texto(_meta(soup, "og:description", "description")),
        "medio": medio,
        "url": normalizar_url(url),
        "fecha": fecha,
        "fuente": "articulo",
    }
    base = FUENTES[medio]["base"]
    enlaces = {
        normalizar_url(urljoin(base, a["href"]))
        for a in soup.find_all("a", href=True)
    }
    enlaces = [u for u in enlaces if urlparse(u).netloc == urlparse(base).netloc and es_articulo(medio, u)]
    return meta, enlaces


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
class Recolector:
    def __init__(self, raw_dir: Path, pausa: float = PAUSA_SEG):
        self.dl = Descargador(raw_dir, pausa=pausa)
        self.candidatos: dict[str, dict] = {}
        self.visitados: set[str] = set()

    # -- fusión: conserva el primer valor no vacío de cada campo --
    def _agregar(self, registros: list[dict]) -> int:
        nuevos = 0
        for r in registros:
            r = dict(r)
            r["seccion_original"] = seccion_desde_url(r["url"])
            previo = self.candidatos.get(r["url"])
            if previo is None:
                self.candidatos[r["url"]] = r
                nuevos += 1
            else:
                for k, v in r.items():
                    if v and not previo.get(k):
                        previo[k] = v
                if r["fuente"] not in previo["fuente"]:
                    previo["fuente"] += "|" + r["fuente"]
        return nuevos

    def recolectar_fuentes(self, verbose: bool = True) -> None:
        for medio, cfg in FUENTES.items():
            pref = "lr" if medio == LR else "ec"
            for sec, url in cfg["rss"].items():
                txt = self.dl.get(url, f"{pref}_rss_{_slug(sec)}.xml")
                n = self._agregar(parsear_rss(txt, medio, sec)) if txt else 0
                if verbose:
                    print(f"[RSS]     {medio:<12} {sec:<26} +{n}")
            for sec, url in cfg["sitemaps"].items():
                txt = self.dl.get(url, f"{pref}_sitemap_{_slug(sec)}.xml")
                n = self._agregar(parsear_sitemap_noticias(txt, medio, sec)) if txt else 0
                if verbose:
                    print(f"[SITEMAP] {medio:<12} {sec:<26} +{n}")
            for sec, url in cfg["html"].items():
                txt = self.dl.get(url, f"{pref}_html_{_slug(sec)}.html.gz")
                n = self._agregar(parsear_html_seccion(txt, medio, sec)) if txt else 0
                if verbose:
                    print(f"[HTML]    {medio:<12} {sec:<26} +{n}")

    def conteo_validos(self, medio: str, etiqueta: str) -> int:
        return sum(
            1 for r in self.candidatos.values()
            if r["medio"] == medio and mapear_etiqueta(medio, r["seccion_original"]) == etiqueta
        )

    def visitar_articulo(self, url: str) -> list[str]:
        medio = LR if "larepublica" in url else EC
        self.visitados.add(url)
        txt = self.dl.get(url)
        if not txt:
            return []
        meta, enlaces = parsear_articulo(txt, medio, url)
        meta["fuente"] = "articulo"
        self._agregar([meta])
        self._agregar([{"titular": "", "bajada": "", "medio": medio, "url": u,
                        "fecha": fecha_desde_url(u), "fuente": "enlace"} for u in enlaces])
        return enlaces

    def rastrear_hasta_cuota(self, cuota: int, max_paginas: int = 60, seed: int = 42,
                             verbose: bool = True) -> None:
        """Rastreo en anchura dentro de la misma categoría hasta alcanzar la cuota."""
        rng = random.Random(seed)
        for medio in FUENTES:
            for etiqueta in ETIQUETAS:
                paginas = 0
                while self.conteo_validos(medio, etiqueta) < cuota and paginas < max_paginas:
                    pendientes = [
                        u for u, r in self.candidatos.items()
                        if r["medio"] == medio and u not in self.visitados
                        and mapear_etiqueta(medio, r["seccion_original"]) == etiqueta
                    ]
                    if not pendientes:
                        break
                    self.visitar_articulo(rng.choice(pendientes))
                    paginas += 1
                if verbose:
                    print(f"[CRAWL]   {medio:<12} {etiqueta:<20} páginas={paginas:>3} "
                          f"candidatos={self.conteo_validos(medio, etiqueta)}")

    def completo(self, url: str) -> bool:
        r = self.candidatos[url]
        return bool(r["titular"] and r["bajada"] and r["fecha"])

    def completar_hasta_cuota(self, cuota: int, seed: int = 42, verbose: bool = True) -> None:
        """Visita artículos incompletos (sin titular/bajada/fecha), en orden aleatorio
        reproducible, hasta tener `cuota` registros completos por medio y categoría."""
        rng = random.Random(seed)
        for medio in FUENTES:
            for etiqueta in ETIQUETAS:
                urls = sorted(
                    u for u, r in self.candidatos.items()
                    if r["medio"] == medio and mapear_etiqueta(medio, r["seccion_original"]) == etiqueta
                )
                rng.shuffle(urls)
                n_completos = sum(self.completo(u) for u in urls)
                visitas = 0
                for u in urls:
                    if n_completos >= cuota:
                        break
                    if self.completo(u) or u in self.visitados:
                        continue
                    self.visitar_articulo(u)
                    visitas += 1
                    n_completos += self.completo(u)
                if verbose:
                    print(f"[META]    {medio:<12} {etiqueta:<20} visitas={visitas:>3} completos={n_completos}")

    def guardar_crudo(self, ruta: Path, fecha_descarga: str) -> None:
        payload = {
            "fecha_descarga": fecha_descarga,
            "user_agent": USER_AGENT,
            "pausa_seg": self.dl.pausa,
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bitacora_requests": self.dl.bitacora,
            "candidatos": list(self.candidatos.values()),
        }
        Path(ruta).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def muestrear_balanceado(df, cuota: int = 50, n_por_medio: int = 200, seed: int = 42):
    """Toma `cuota` noticias por (medio, etiqueta), priorizando registros con bajada.
    Si una celda no llega a la cuota, completa el medio con noticias de las otras
    categorías para mantener exactamente `n_por_medio` filas por medio."""
    import pandas as pd

    partes = []
    for _, g in df.groupby("medio", sort=True):
        g = g.sample(frac=1, random_state=seed)
        g = g.assign(_completo=g["bajada"].str.len() > 0).sort_values(
            "_completo", ascending=False, kind="stable")
        elegidos = g.groupby("etiqueta", group_keys=False).head(cuota)
        faltan = n_por_medio - len(elegidos)
        if faltan > 0:
            elegidos = pd.concat([elegidos, g.drop(elegidos.index).head(faltan)])
        partes.append(elegidos.head(n_por_medio))
    return pd.concat(partes).drop(columns="_completo").reset_index(drop=True)
