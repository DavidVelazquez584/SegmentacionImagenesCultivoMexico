# ==============================================================================
# Fase 2: benchmark de velocidad y memoria de TSViT en la GPU local (bf16).
# Mide: (1) iteraciones de entrenamiento (adelante + atrás + AdamW) con la pérdida del Avance 6,
#       (2) lectura de parches reales desde los .npy, (3) inferencia de validación con ventanas
#       solapadas, y extrapola el tiempo por época y total.
# ==============================================================================
import os, sys, json, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsvit_modelo import TSViTDias, CONFIG_TSVIT, CONFIG_TSVIT_S

PASTIS = r"C:\Users\lalo4\Downloads\PASTIS\PASTIS"
DEVICE = torch.device("cuda")
NC, VOID = 19, 19
N_ITER, N_WARM = int(os.environ.get("N_ITER", 150)), 10
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prop = torch.cuda.get_device_properties(0)
print(f"GPU: {prop.name} | VRAM {prop.total_memory / 1024 ** 3:.1f} GB | torch {torch.__version__} | bf16 "
      f"{torch.cuda.is_bf16_supported()}")

g = json.load(open(os.path.join(PASTIS, "metadata.geojson")))
meta = [f["properties"] for f in g["features"]]
ids_train = sorted(int(m["ID_PATCH"]) for m in meta if int(m["Fold"]) in (1, 2, 3))
ids_val = sorted(int(m["ID_PATCH"]) for m in meta if int(m["Fold"]) == 4)
pesos = torch.ones(NC)  # la velocidad no depende de los valores de los pesos
ce = nn.CrossEntropyLoss(weight=pesos.to(DEVICE), ignore_index=VOID)


def dice_loss(logits, y, eps=1e-6):
    valid = (y != VOID)
    probs = logits.softmax(dim=1) * valid.unsqueeze(1)
    onehot = F.one_hot(y.clamp(max=NC - 1), NC).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    presente = onehot.sum(dim=(0, 2, 3)) > 0
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice[presente].mean()


def bench_train(cfg, nombre, B, T=60):
    torch.manual_seed(0)
    m = TSViTDias(cfg).to(DEVICE).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    x = torch.randn(B, T, 24, 24, 10, device=DEVICE)
    d = torch.sort(torch.randint(16, 422, (B, T), device=DEVICE), dim=1).values
    y = torch.randint(0, NC + 1, (B, 24, 24), device=DEVICE)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(N_ITER + N_WARM):
            if i == N_WARM:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x, d)
            logits = logits.float()
            loss = ce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        torch.cuda.synchronize()
        seg = (time.time() - t0) / N_ITER
        pico = torch.cuda.max_memory_allocated() / 1024 ** 3
    except torch.cuda.OutOfMemoryError:
        seg, pico = None, None
    n = sum(p.numel() for p in m.parameters())
    del m, opt; torch.cuda.empty_cache()
    return {"modelo": nombre, "params": n, "B": B, "T": T, "s_iter": seg, "GB": pico}


print("\n=== (1) Entrenamiento: adelante + atrás + AdamW, bf16, T=60 ===")
res = []
for cfg, nombre in ((CONFIG_TSVIT, "TSViT"), (CONFIG_TSVIT_S, "TSViT-S")):
    for B in (16, 8):
        r = bench_train(cfg, nombre, B); res.append(r)
        print(f"{nombre:8s} params {r['params']:,} | B={B:2d} | "
              + (f"{r['s_iter'] * 1000:6.1f} ms/iter | pico {r['GB']:.2f} GB" if r["s_iter"] else "NO CABE"), flush=True)

print("\n=== (2) Lectura de parches reales (np.load completo, int16) ===")
rng = np.random.default_rng(0)
muestra = rng.choice(ids_train, 20, replace=False)
t0 = time.time(); tam = 0
for pid in muestra:
    a = np.load(os.path.join(PASTIS, "DATA_S2", f"S2_{pid}.npy")); tam += a.nbytes
    yb = np.load(os.path.join(PASTIS, "ANNOTATIONS", f"TARGET_{pid}.npy"))
t_lect = (time.time() - t0) / len(muestra)
print(f"{t_lect * 1000:.0f} ms por parche | {tam / len(muestra) / 1e6:.1f} MB por parche | "
      f"train completo en RAM ≈ {tam / len(muestra) * len(ids_train) / 1e9:.1f} GB")

print("\n=== (3) Inferencia de validación: 36 ventanas solapadas por parche, bf16 ===")
m = TSViTDias(CONFIG_TSVIT).to(DEVICE).eval()
t_inf = []
with torch.no_grad():
    for pid in muestra[:10]:
        a = torch.from_numpy(np.load(os.path.join(PASTIS, "DATA_S2", f"S2_{pid}.npy")).astype(np.float32))
        T = a.shape[0]
        x = torch.zeros(60, 128, 128, 10); x[:min(T, 60)] = a[:60].permute(0, 2, 3, 1)
        d = torch.zeros(1, 60, dtype=torch.long); d[0, :min(T, 60)] = torch.arange(min(T, 60)) * 7 + 16
        torch.cuda.synchronize(); t0 = time.time()
        offs = [0, 21, 42, 62, 83, 104]
        v = torch.stack([x[:, r:r + 24, c:c + 24] for r in offs for c in offs]).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(v, d.expand(len(v), -1).to(DEVICE)).float().softmax(1)
        torch.cuda.synchronize(); t_inf.append(time.time() - t0)
t_val = np.mean(t_inf[2:]) + t_lect
print(f"{np.mean(t_inf[2:]) * 1000:.0f} ms de GPU + {t_lect * 1000:.0f} ms de lectura por parche "
      f"→ validación completa (482 parches) ≈ {t_val * len(ids_val) / 60:.1f} min")

print("\n=== Extrapolación (25 ventanas por parche de entrenamiento y época) ===")
ventanas = 25 * len(ids_train)
for r in res:
    if not r["s_iter"]:
        continue
    pasos = ventanas // r["B"]
    t_ep = pasos * r["s_iter"] + len(ids_train) * t_lect + t_val * len(ids_val)
    print(f"{r['modelo']:8s} B={r['B']:2d}: {pasos} iter/época | ≈ {t_ep / 60:.1f} min/época | "
          f"100 épocas ≈ {t_ep * 100 / 3600:.1f} h")
