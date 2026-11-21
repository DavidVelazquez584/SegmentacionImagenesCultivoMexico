# (1) Tiempo por parche en GPU y CPU, con y sin TTA; acuerdo CPU fp32 vs GPU bf16.
# (2) Verificación de resumen_parche: hectáreas que suman 163.84 e IoU por parche coherente.
# Uso: python prueba_tiempos_resumen.py
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inferencia as inf  # noqa: E402

meta = inf.MetadatosPASTIS(inf.PASTIS_ROOT / "metadata.geojson")
ids = sorted(p for p, f in meta.fold.items() if f == 5)
muestra = ids[::165][:3]                       # 3 parches repartidos en el fold 5
datos = {p: (np.load(inf.PASTIS_ROOT / "DATA_S2" / f"S2_{p}.npy"), meta.dias(p),
             np.load(inf.PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{p}.npy")[0].astype(np.int64)) for p in muestra}
print("Parches de muestra:", muestra, "| fechas:", [len(datos[p][1]) for p in muestra])
print("Hilos de CPU de torch:", torch.get_num_threads())

resultados, preds = [], {}
for disp in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
    seg = inf.Segmentador(disp)
    seg.probabilidades(*datos[muestra[0]][:2])            # calentamiento
    for tta in (False, True):
        ts = []
        for p in muestra:
            if disp == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            prob = seg.probabilidades(*datos[p][:2], tta=tta)
            if disp == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
            preds[(disp, tta, p)] = prob
        resultados.append((disp, tta, np.mean(ts)))
        print(f"{disp:4s} | TTA {'sí' if tta else 'no'} | {np.mean(ts):6.2f} s/parche (solo modelo)", flush=True)
    del seg

if ("cuda", True, muestra[0]) in preds:
    for tta in (False, True):
        acuerdo = np.mean([(preds[("cuda", tta, p)].argmax(0) == preds[("cpu", tta, p)].argmax(0)).mean() for p in muestra])
        dmax = max(np.abs(preds[("cuda", tta, p)] - preds[("cpu", tta, p)]).max() for p in muestra)
        print(f"Acuerdo de píxeles CPU fp32 vs GPU bf16 (TTA {'sí' if tta else 'no'}): {acuerdo:.5f} | "
              f"máx |Δprob| {dmax:.4f}")

print("\n=== Verificación de resumen_parche ===")
disp = "cuda" if torch.cuda.is_available() else "cpu"
for p in muestra:
    prob = preds[(disp, True, p)]
    y = datos[p][2]
    df = inf.resumen_parche(prob, y)
    s_pred = df["ha_pred"].sum()
    s_real = df["ha_real"].sum() + df.attrs["ha_sin_etiqueta"]
    assert abs(s_pred - inf.HA_PARCHE) < 0.05, s_pred
    assert abs(s_real - inf.HA_PARCHE) < 0.05, s_real
    # IoU por parche contra el cálculo con matriz de confusión
    pred = prob.argmax(0); m = y != inf.VOID_LABEL
    conf = np.bincount(y[m] * 19 + pred[m], minlength=361).reshape(19, 19)
    tp = np.diag(conf); union = conf.sum(0) + conf.sum(1) - tp
    for _, f in df.iterrows():
        c = f["id"]
        if union[c] > 0:
            assert abs(f["iou_parche"] - tp[c] / union[c]) < 1e-3, (p, c)
    top = df.sort_values("ha_pred", ascending=False).head(3)
    print(f"Parche {p} (fold {meta.fold[p]}): Σ ha pred {s_pred:.2f} | Σ ha real + sin etiqueta {s_real:.2f} | "
          f"confianza global {df.attrs['confianza_global']}")
    print(top[["clase", "ha_pred", "ha_real", "pct_parche", "confianza_media", "iou_parche"]].to_string(index=False))
print("\nresumen_parche: VERIFICADO (hectáreas suman 163.84 e IoU coincide con la matriz de confusión)")
