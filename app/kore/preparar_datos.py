# Crea chatbot/datos/ con lo mínimo que Koré necesita, para no depender del PASTIS completo (29 GB):
#   - metadatos.json: fold y fechas de los 2,433 parches (para segmentar cualquier parche subido);
#   - etiquetas_fold5.npz: etiqueta semántica (uint8) de los 496 parches del fold 5 (prueba);
#   - ejemplos/S2_<ID>.npy: las series Sentinel-2 de los parches de la galería.
# PASTIS: CC BY 4.0 (Sainte Fare Garnot y Landrieu, 2021) — se redistribuye un subconjunto con atribución.
# Uso: python preparar_datos.py
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import inferencia as inf  # noqa: E402

EJEMPLOS = [40173, 10530, 20581, 40181, 40305, 10101, 30024, 20392]
SALIDA = AQUI / "datos"
(SALIDA / "ejemplos").mkdir(parents=True, exist_ok=True)

meta = inf.MetadatosPASTIS(inf.PASTIS_ROOT / "metadata.geojson")
compacto = {str(p): {"fold": meta.fold[p], "fechas": [f.strftime("%Y%m%d") for f in meta.fechas[p]]}
            for p in sorted(meta.fold)}
(SALIDA / "metadatos.json").write_text(json.dumps(compacto, separators=(",", ":")), encoding="utf-8")

etiquetas = {}
for p in sorted(p for p, f in meta.fold.items() if f == 5):
    y = np.load(inf.PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{p}.npy")[0]
    assert y.min() >= 0 and y.max() <= 19
    etiquetas[str(p)] = y.astype(np.uint8)
np.savez_compressed(SALIDA / "etiquetas_fold5.npz", **etiquetas)

for p in EJEMPLOS:
    assert meta.fold[p] == 5, f"{p} no es del fold 5"
    shutil.copy(inf.PASTIS_ROOT / "DATA_S2" / f"S2_{p}.npy", SALIDA / "ejemplos" / f"S2_{p}.npy")

for f in sorted(SALIDA.rglob("*")):
    if f.is_file():
        print(f"{f.relative_to(AQUI)}: {f.stat().st_size / 1e6:.1f} MB")
print(f"Parches en metadatos: {len(compacto)} | etiquetas del fold 5: {len(etiquetas)} | ejemplos: {len(EJEMPLOS)}")
