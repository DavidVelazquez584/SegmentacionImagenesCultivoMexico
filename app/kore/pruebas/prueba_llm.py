# Conjunto de prueba del chatbot: 10 preguntas sobre 3 parches del fold 5.
# Cada respuesta se verifica automáticamente (¿contiene el número exacto de la tabla? ¿advierte la
# clase débil? ¿dice que no sabe?) y además se guarda completa para revisión humana.
# Uso: python prueba_llm.py <backend> [modelo]     p. ej.  python prueba_llm.py ollama qwen2.5:3b
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inferencia as inf  # noqa: E402
from contexto import GRUPOS, construir_contexto  # noqa: E402
from llm import BackendOllama, obtener_backend  # noqa: E402

tipo = sys.argv[1] if len(sys.argv) > 1 else "ollama"
llm = BackendOllama(modelo=sys.argv[2]) if (tipo == "ollama" and len(sys.argv) > 2) else obtener_backend(tipo)

meta = inf.MetadatosPASTIS(inf.PASTIS_ROOT / "metadata.geojson")
seg = inf.Segmentador()
CACHE = Path(__file__).with_name("cache_prob")
CACHE.mkdir(exist_ok=True)


def preparar(pid):
    f = CACHE / f"{pid}.npy"
    if f.is_file():
        prob = np.load(f)
    else:
        prob = seg.probabilidades(np.load(inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid}.npy"), meta.dias(pid), tta=False)
        np.save(f, prob)
    y = np.load(inf.PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].astype(np.int64)
    df = inf.resumen_parche(prob, y)
    fs = meta.fechas[pid]
    info = {"id": pid, "fold": meta.fold[pid], "n_fechas": len(fs), "fecha_ini": fs[0].strftime("%d/%m/%Y"),
            "fecha_fin": fs[-1].strftime("%d/%m/%Y"), "tiene_real": True, "tta": False}
    return df, construir_contexto(df, info)


def fila(df, nombre):
    return df[df["clase"] == nombre].iloc[0]


def grupo(df, prefijo, col="ha_pred"):
    ids = next(v for k, v in GRUPOS.items() if k.startswith(prefijo))
    return df[df["id"].isin(ids)][col].sum()


def contiene_numero(texto, x, dec=2):
    s = f"{x:.{dec}f}"
    return s in texto or s.replace(".", ",") in texto


NO_SABE = re.compile(r"no (lo )?s[ée]|no (tengo|tiene|dispongo|cuento)|no (es posible|puedo)|no est[áa] disponible|"
                     r"fuera de|no incluye|no contiene|no hay (informaci[óo]n|datos)", re.I)
INVENTA_DINERO = re.compile(r"\$\s?\d|\d\s?(usd|mxn|eur|€|pesos|d[óo]lares)|\d\s?(t|ton|toneladas?)\s?(/|por)\s?ha", re.I)
ADVIERTE = re.compile(r"d[ée]bil|poco confiable|baja confianza|precauci[óo]n|cuidado|incertidumbre|menos confiable", re.I)

casos = []
df1, s1 = preparar(40173)
df2, s2 = preparar(10530)
df3, s3 = preparar(20392)
casos += [
    (40173, s1, "¿Cuántas hectáreas son de maíz?",
     lambda r: contiene_numero(r, fila(df1, "Maíz")["ha_pred"]), "número exacto de ha de maíz"),
    (40173, s1, "Sin contar el fondo, ¿qué cultivo domina el parche?",
     lambda r: "maíz" in r.lower(), "nombra el maíz"),
    (40173, s1, "¿Cuántas hectáreas de oleaginosas hay?",
     lambda r: contiene_numero(r, grupo(df1, "Oleaginosas")), "total de oleaginosas"),
    (40173, s1, "¿Cuál es el precio del maíz por tonelada en Francia?",
     lambda r: bool(NO_SABE.search(r)), "dice que no lo sabe"),
    (10530, s2, "¿Cuántas hectáreas de trigo blando de invierno predice el modelo y cuántas hay en realidad?",
     lambda r: contiene_numero(r, fila(df2, "Trigo blando de invierno")["ha_pred"]) and
               contiene_numero(r, fila(df2, "Trigo blando de invierno")["ha_real"]), "ha predichas y reales"),
    (10530, s2, "¿Qué tan seguro está el modelo en este parche?",
     lambda r: contiene_numero(r, df2.attrs["confianza_global"], 3), "confianza global exacta"),
    (10530, s2, "¿Cuántas hectáreas de cereales de grano hay?",
     lambda r: contiene_numero(r, grupo(df2, "Cereales")), "total de cereales"),
    (20392, s3, "¿Cuántas hectáreas de frutas, hortalizas y flores hay?",
     lambda r: contiene_numero(r, fila(df3, "Frutas, hortalizas y flores")["ha_pred"]) and bool(ADVIERTE.search(r)),
     "número exacto + advertencia de clase débil"),
    (20392, s3, "¿Qué IoU tiene el trigo duro de invierno en este parche?",
     lambda r: contiene_numero(r, fila(df3, "Trigo duro de invierno")["iou_parche"], 3), "IoU exacto"),
    (20392, s3, "¿Cuánta lluvia cayó en esta zona en 2019?",
     lambda r: bool(NO_SABE.search(r)), "dice que no lo sabe"),
    # Caso real reportado por el usuario: el modelo inventó precios y rendimientos.
    (40173, s1, "Si vendiera todo el maíz, ¿cuánto dinero podría generar en México?",
     lambda r: not INVENTA_DINERO.search(r), "no inventa precios, rendimientos ni montos"),
]

print(f"Backend: {llm.nombre}\n")
resultados, t_total = [], 0.0
for i, (pid, sistema, pregunta, ok, criterio) in enumerate(casos, 1):
    t = time.perf_counter()
    try:
        r = llm.responder(sistema, [{"role": "user", "content": pregunta}])
    except Exception as e:
        r = f"[ERROR] {e}"
    dt = time.perf_counter() - t
    t_total += dt
    # Una respuesta que vuelca toda la tabla "contiene" cualquier número: se exige además que sea breve.
    paso = ok(r) and len(r) <= 1200
    resultados.append({"n": i, "parche": pid, "pregunta": pregunta, "criterio": criterio, "pasa": paso,
                       "segundos": round(dt, 1), "respuesta": r})
    print(f"{'✔' if paso else '✘'} {i:2d}. [{pid}] {pregunta}  ({dt:.1f} s)\n     criterio: {criterio}\n"
          f"     respuesta: {r[:300].replace(chr(10), ' ')}\n")

n_ok = sum(r["pasa"] for r in resultados)
print(f"RESULTADO: {n_ok}/{len(resultados)} | tiempo medio {t_total / len(resultados):.1f} s por respuesta")
if hasattr(llm, "uso"):
    u = llm.uso
    print(f"Tokens: entrada {u['entrada']:,} | leídos de caché {u['cache_leido']:,} | salida {u['salida']:,}")
nombre = re.sub(r"[^A-Za-z0-9]+", "_", llm.nombre).strip("_")
salida = Path(__file__).with_name(f"resultado_llm_{nombre}.json")
salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
print("Respuestas completas en", salida)
