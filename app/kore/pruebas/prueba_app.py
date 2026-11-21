# Prueba de la lógica de app.py sin navegador: anti-fuga, errores de entrada y chat sin parche.
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402  (no lanza el servidor: launch está bajo __main__)

meta = app.META
tmp = Path(tempfile.mkdtemp())


def _valor(x):
    """Las imágenes pueden volver como gr.Image(value=..., label=...): se compara su valor."""
    return getattr(x, "value", x) if not isinstance(x, np.ndarray) else x


def caso(nombre, salida):
    salida = tuple(_valor(v) if i in (0, 1, 2) else v for i, v in enumerate(salida))
    rgb, pred, real, ley, tabla, estado, chat, msg, kpis = salida
    assert "kore-kpis" in kpis or "kore-vacio" in kpis
    print(f"--- {nombre}\n  mapa predicho: {pred is not None} | etiqueta real: {real is not None} | "
          f"columnas: {list(tabla.columns) if tabla is not None else None}\n  aviso: {msg[:160]}")
    return salida


# 1) Parche del fold 1 subido como archivo: debe segmentarse SIN etiqueta real ni IoU.
pid1 = min(p for p, f in meta.fold.items() if f == 1)
s = caso(f"fold 1 (ID {pid1})", app.analizar(None, str(app.inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid1}.npy"), False))
assert s[1] is not None and s[2] is None and "IoU del parche" not in s[4].columns
assert "fold 1" in s[7] and "aciertos" in s[5]["sistema"] and "NO hables" in s[5]["sistema"]

# 2) Parche del fold 4 (validación): igual que el anterior.
pid4 = min(p for p, f in meta.fold.items() if f == 4)
s = caso(f"fold 4 (ID {pid4})", app.analizar(None, str(app.inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid4}.npy"), False))
assert s[2] is None and "Hectáreas (real)" not in s[4].columns

# 3) Archivo con nombre inválido
mal = tmp / "mi_parcela.npy"; shutil.copy(app.inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid1}.npy", mal)
s = caso("nombre inválido", app.analizar(None, str(mal), False)); assert s[1] is None and "S2_" in s[7]

# 4) ID que no existe en PASTIS
inex = tmp / "S2_99999999.npy"; np.save(inex, np.zeros((5, 10, 128, 128), np.int16))
s = caso("ID inexistente", app.analizar(None, str(inex), False)); assert s[1] is None and "no existe" in s[7]

# 5) Forma incorrecta con ID válido
pid5 = 40173
forma = tmp / f"S2_{pid5}.npy"; np.save(forma, np.zeros((3, 10, 64, 64), np.int16))
s = caso("forma incorrecta", app.analizar(None, str(forma), False)); assert s[1] is None and "forma" in s[7]

# 5b) Parche del fold 5 que NO está en la galería, subido como archivo: debe traer etiqueta real
pid_f5 = next(p for p, f in sorted(meta.fold.items()) if f == 5 and p not in app.EJEMPLOS)
s = caso(f"fold 5 subido (ID {pid_f5})", app.analizar(None, str(app.inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid_f5}.npy"), False))
assert s[2] is not None and "IoU del parche" in s[4].columns

# 6) Ejemplo del fold 5: con etiqueta real; el prompt no debe tener 'nan'
s = caso("ejemplo fold 5", app.analizar(app.OPCIONES[0], None, False))
assert s[2] is not None and "IoU del parche" in s[4].columns and "nan" not in s[5]["sistema"]

# 7) Chat sin parche analizado
_, h = app.conversar("¿cuánto maíz hay?", [], None)
print("--- chat sin parche:", h[-1]["content"]); assert "Primero" in h[-1]["content"]
print("\nprueba_app: TODAS LAS VERIFICACIONES PASARON")

# 8) La foto en color verdadero no debe salir en blanco: en 40181 y 40305 la fecha central está nublada.
for pid in (40181, 40305):
    x = np.load(app.DATOS / "ejemplos" / f"S2_{pid}.npy")
    t = app.fecha_despejada(x)
    img = app.rgb_verdadero(x, t)
    saturado = float((img >= 250).all(axis=2).mean())
    print(f"--- foto {pid}: fecha {t} de {len(x)} | píxeles blancos {saturado:.1%}")
    assert saturado < 0.10
print("\nfoto en color verdadero: CORRECTA")
