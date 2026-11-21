# ==============================================================================
# Fase 1: evaluación de los pesos oficiales de TSViT (fold 1 = train 1-3 / val 4 / test 5)
# con la definición de métricas del Avance 6 (19 clases, Void ignorado, Fondo incluido).
# Uso: python fase1_tsvit_oficial.py <fold> <L> [modos...]
#   fold : 4 o 5          L : longitud máxima de la serie (60 = config de eval, 61 = config de test)
# ==============================================================================
import os, sys, json, time, datetime
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsvit_modelo import TSViT, CONFIG_TSVIT, MEAN_OFICIAL, STD_OFICIAL

PASTIS = r"C:\Users\lalo4\Downloads\PASTIS\PASTIS"
CKPT = (r"C:\Users\lalo4\Downloads\Avances\externos\TSViT_PASTIS24_oficial\root\streaming-inference\temp"
        r"\personal\tsvit\PASTIS24_checkpoints\TSViT_fold1\best.pth")
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_fase1")
NUM_CLASSES, VOID = 19, 19
DEVICE = torch.device("cuda")

fold, L = int(sys.argv[1]), int(sys.argv[2])
modos = sys.argv[3:] or ["rejilla", "solape"]
os.makedirs(SALIDA, exist_ok=True)

g = json.load(open(os.path.join(PASTIS, "metadata.geojson")))
meta = [f["properties"] for f in g["features"]]
ids = sorted(int(m["ID_PATCH"]) for m in meta if int(m["Fold"]) == fold)
if os.environ.get("MAXP"):
    ids = ids[:int(os.environ["MAXP"])]          # solo para pruebas rápidas
fechas = {int(m["ID_PATCH"]): (json.loads(m["dates-S2"]) if isinstance(m["dates-S2"], str) else m["dates-S2"]) for m in meta}


def doy_de(pid):
    d = fechas[pid]
    return np.array([datetime.datetime.strptime(str(d[k]), "%Y%m%d").timetuple().tm_yday
                     for k in sorted(d, key=int)])


MEAN = torch.tensor(MEAN_OFICIAL, dtype=torch.float32).view(1, 10, 1, 1)
STD = torch.tensor(STD_OFICIAL, dtype=torch.float32).view(1, 10, 1, 1)


def preparar(pid):
    """Réplica del pipeline oficial sobre el parche completo: orden por DOY, Normalize, TileDates,
    CutOrPad(from_start) y ToTHWC. Devuelve (L, 128, 128, 11) y la etiqueta (128, 128)."""
    x = np.load(os.path.join(PASTIS, "DATA_S2", f"S2_{pid}.npy"))
    y = np.load(os.path.join(PASTIS, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0].astype(np.int64)
    doy = doy_de(pid)
    idx = np.argsort(doy)                                   # igual que data2windows.py
    x, doy = x[idx], doy[idx]
    xt = (torch.tensor(x).to(torch.float32) - MEAN) / STD
    dt = torch.tensor(doy).to(torch.float32) / 365.0001
    dt = dt.view(-1, 1, 1, 1).expand(-1, 1, 128, 128)
    xt = torch.cat([xt, dt], dim=1)                          # T x 11 x H x W
    T = xt.shape[0]
    if T < L:
        xt = torch.cat([xt, torch.zeros((L - T,) + tuple(xt.shape[1:]))], dim=0)
    elif T > L:
        xt = xt[:L]
    return xt.permute(0, 2, 3, 1).contiguous(), torch.from_numpy(y)


OFFSETS = {"rejilla": [0, 24, 48, 72, 96],       # ventanas de PASTIS24 (sin solape, sin bordes 120-127)
           "solape": [0, 21, 42, 62, 83, 104]}   # 6 ventanas por eje: cubre los 128 px con solape de 3-4 px


@torch.no_grad()
def predecir(model, xt, offs):
    """Promedia las probabilidades de las ventanas que cubren cada píxel."""
    ventanas = torch.stack([xt[:, r:r + 24, c:c + 24, :] for r in offs for c in offs]).to(DEVICE)
    logits = model(ventanas).float()                         # n x 19 x 24 x 24
    prob = logits.softmax(1)
    acum = torch.zeros(NUM_CLASSES, 128, 128, device=DEVICE)
    cuenta = torch.zeros(1, 128, 128, device=DEVICE)
    k = 0
    for r in offs:
        for c in offs:
            acum[:, r:r + 24, c:c + 24] += prob[k]; cuenta[:, r:r + 24, c:c + 24] += 1; k += 1
    cubierto = (cuenta[0] > 0)
    return (acum / cuenta.clamp(min=1)).argmax(0), cubierto


def confusion(pred, y, mascara):
    m = mascara & (y != VOID)
    idx = y[m] * NUM_CLASSES + pred[m]
    return torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES).cpu().numpy()


def metricas(conf):
    conf = conf.astype(np.float64)
    tp = np.diag(conf); real = conf.sum(1); pred = conf.sum(0); union = real + pred - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), 0.0)
    f1 = np.where(real + pred > 0, 2 * tp / np.maximum(real + pred, 1), 0.0)
    return {"Accuracy": tp.sum() / max(conf.sum(), 1), "mIoU": iou.mean(), "Macro F1": f1.mean(), "iou": iou, "f1": f1}


LEGACY = os.environ.get("LEGACY", "1") == "1"
model = TSViT(dict(CONFIG_TSVIT, legacy_reshape=LEGACY)).to(DEVICE)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
faltan = model.load_state_dict(sd, strict=True)
model.eval()
print(f"Pesos cargados (strict=True) | parámetros: {sum(p.numel() for p in model.parameters()):,} | "
      f"fold {fold} ({len(ids)} parches) | L={L} | modos {modos} | legacy_reshape={LEGACY}")

region120 = torch.zeros(128, 128, dtype=torch.bool); region120[:120, :120] = True
completo = torch.ones(128, 128, dtype=torch.bool)
por_parche = {f"{m}|{r}": np.zeros((len(ids), NUM_CLASSES, NUM_CLASSES), np.int64)
              for m in modos for r in ("120", "128")}
t0 = time.time()
for i, pid in enumerate(ids):
    xt, y = preparar(pid)
    y = y.to(DEVICE)
    for m in modos:
        pred, cub = predecir(model, xt, OFFSETS[m])
        por_parche[f"{m}|120"][i] = confusion(pred, y, region120.to(DEVICE))
        if bool(cub.all()):
            por_parche[f"{m}|128"][i] = confusion(pred, y, completo.to(DEVICE))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(ids)} parches | {time.time() - t0:.0f} s", flush=True)

print(f"Tiempo total: {time.time() - t0:.0f} s")
for clave, cp in por_parche.items():
    if cp.sum() == 0:
        continue
    r = metricas(cp.sum(0))
    print(f"fold {fold} | L={L} | {clave:14s} | Acc {r['Accuracy']:.4f} | Macro F1 {r['Macro F1']:.4f} | mIoU {r['mIoU']:.4f}")
np.savez_compressed(os.path.join(SALIDA, f"conf_fold{fold}_L{L}.npz"), ids=np.array(ids),
                    **{k.replace("|", "_"): v for k, v in por_parche.items()})
