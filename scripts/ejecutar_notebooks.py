"""Ejecuta los notebooks en orden y guarda sus salidas en el mismo archivo.

Uso (desde la carpeta tarea_1/):
    python scripts/ejecutar_notebooks.py            # ambos notebooks
    python scripts/ejecutar_notebooks.py 02         # solo el que contenga "02" en el nombre

Equivale a:
    jupyter nbconvert --to notebook --execute --inplace 01_recoleccion.ipynb
    jupyter nbconvert --to notebook --execute --inplace 02_clasificacion.ipynb
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_recoleccion.ipynb", "02_clasificacion.ipynb"]

filtro = sys.argv[1] if len(sys.argv) > 1 else ""
for nombre in NOTEBOOKS:
    if filtro not in nombre:
        continue
    ruta = RAIZ / nombre
    print(f"▶ Ejecutando {nombre} …", flush=True)
    t0 = time.perf_counter()
    nb = nbformat.read(ruta, as_version=4)
    NotebookClient(nb, timeout=7200, kernel_name="python3",
                   resources={"metadata": {"path": str(RAIZ)}}).execute()
    nbformat.write(nb, ruta)
    print(f"  ✔ {nombre} listo en {time.perf_counter() - t0:.0f} s")
