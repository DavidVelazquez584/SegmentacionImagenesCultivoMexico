# Prueba de equivalencia: el módulo chatbot/inferencia.py debe reproducir las métricas del
# notebook del Avance 7 en el fold 5 (TSViT desde cero, TTA): mIoU 0.6587, F1 0.7807, Acc 0.8378.
# Además compara parche por parche contra las matrices guardadas en confusiones_prueba.npz.
# Uso: python prueba_equivalencia.py [cuda|cpu] [n_parches]
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inferencia as inf  # noqa: E402

dispositivo = sys.argv[1] if len(sys.argv) > 1 else "cuda"
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else None

NPZ = Path(r"C:\Users\lalo4\Downloads\Avances\salidas_tsvit\confusiones_prueba.npz")
ref = np.load(NPZ)
ids_ref = ref["ids"].tolist()
conf_ref = ref["TSViT__desde_cero__128"]

meta = inf.MetadatosPASTIS(inf.PASTIS_ROOT / "metadata.geojson")
ids = sorted(p for p, f in meta.fold.items() if f == 5)
assert ids == ids_ref, "Los IDs del fold 5 no coinciden con los del notebook"
if n_max:
    ids = ids[:n_max]

seg = inf.Segmentador(dispositivo)
print(f"Dispositivo: {seg.device} | precisión: {seg.amp or 'fp32'} | checkpoint época {seg.epoca}")


def confusion(pred, y):
    m = y != inf.VOID_LABEL
    return np.bincount(y[m] * inf.NUM_CLASSES + pred[m], minlength=inf.NUM_CLASSES ** 2).reshape(19, 19)


def metricas(conf):
    conf = conf.astype(np.float64)
    tp = np.diag(conf); real = conf.sum(1); pr = conf.sum(0)
    iou = np.where(real + pr - tp > 0, tp / np.maximum(real + pr - tp, 1), 0)
    f1 = np.where(real + pr > 0, 2 * tp / np.maximum(real + pr, 1), 0)
    return tp.sum() / conf.sum(), iou.mean(), f1.mean()


conf = np.zeros((len(ids), 19, 19), np.int64)
iguales, dif_px, t0 = 0, [], time.time()
for i, pid in enumerate(ids):
    x = np.load(inf.PASTIS_ROOT / "DATA_S2" / f"S2_{pid}.npy")
    y = np.load(inf.PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{pid}.npy")[0].astype(np.int64)
    prob = seg.probabilidades(x, meta.dias(pid), tta=True)
    conf[i] = confusion(prob.argmax(0), y)
    j = ids_ref.index(pid)
    iguales += int(np.array_equal(conf[i], conf_ref[j]))
    dif_px.append(int(np.abs(conf[i] - conf_ref[j]).sum() // 2))
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(ids)} parches | {(time.time() - t0) / (i + 1):.2f} s/parche", flush=True)

acc, miou, f1 = metricas(conf.sum(0))
idx_ref = [ids_ref.index(p) for p in ids]
acc_r, miou_r, f1_r = metricas(conf_ref[idx_ref].sum(0))
print(f"\nMódulo:   Acc {acc:.4f} | Macro F1 {f1:.4f} | mIoU {miou:.4f}")
print(f"Notebook: Acc {acc_r:.4f} | Macro F1 {f1_r:.4f} | mIoU {miou_r:.4f}  (mismos {len(ids)} parches)")
print(f"Parches con matriz idéntica: {iguales}/{len(ids)} | píxeles distintos: total {sum(dif_px)}, "
      f"máximo por parche {max(dif_px)}")
print(f"Tiempo medio: {(time.time() - t0) / len(ids):.2f} s/parche (incluye lectura del disco)")
