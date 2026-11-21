# ==============================================================================
# Inferencia de TSViT (entrenado desde cero, Avance 7) para el chatbot de cultivos.
#
# Reproduce EXACTAMENTE el preprocesamiento y la inferencia del notebook
# Avance7_TSViT_vs_UTAE.ipynb (celdas 1.3, 2.2 y 4.2):
#   - días desde 2018-09-01 en orden cronológico;
#   - si el parche tiene más de 60 fechas se toman las 60 primeras (inferencia determinista);
#   - normalización por banda (media/desv. del Avance 6, guardadas en el checkpoint);
#   - 36 ventanas de 24×24 solapadas (offsets 0, 21, 42, 62, 83, 104) promediando probabilidades;
#   - TTA opcional de 8 transformaciones (rotaciones × volteo), como se eligió en validación.
# En GPU usa bf16/fp16 (autocast, igual que el notebook); en CPU usa fp32.
# ==============================================================================
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "modelo"))
from tsvit_modelo import TSViTDias  # noqa: E402

# ------------------------------------------------------------------ constantes
NUM_CLASSES = 19
VOID_LABEL = 19
MAX_SEQ_LEN = 60
VENTANA = 24
OFFSETS_SOLAPE = [0, 21, 42, 62, 83, 104]
REFERENCE_DATE = datetime(2018, 9, 1)
HA_POR_PIXEL = 0.01                    # píxel Sentinel-2 de 10 m × 10 m = 100 m²
HA_PARCHE = 128 * 128 * HA_POR_PIXEL   # 163.84 ha
LOTE_INFERENCIA = 12
TRANSFORMACIONES = [(k, f) for k in range(4) for f in (False, True)]

NOMBRES_CLASES = {
    0: "Fondo", 1: "Pradera", 2: "Trigo blando de invierno", 3: "Maíz",
    4: "Cebada de invierno", 5: "Colza de invierno", 6: "Cebada de primavera",
    7: "Girasol", 8: "Vid", 9: "Remolacha", 10: "Triticale de invierno",
    11: "Trigo duro de invierno", 12: "Frutas, hortalizas y flores", 13: "Papa",
    14: "Forraje leguminoso", 15: "Soya", 16: "Huerto", 17: "Cereal mixto",
    18: "Sorgo",
}
# Clases con IoU bajo en el fold 5 (0.37–0.47): las predicciones de estas clases son poco confiables.
CLASES_DEBILES = {12, 14, 17, 18}

RUTA_PESOS = Path(os.environ.get("TSVIT_PESOS", AQUI / "modelo" / "tsvit_best.pt"))
PASTIS_ROOT = Path(os.environ.get("PASTIS_ROOT", r"C:\Users\lalo4\Downloads\PASTIS\PASTIS"))


# ------------------------------------------------------------------ metadatos de PASTIS
def _fechas_ordenadas(valor):
    if isinstance(valor, str):
        valor = json.loads(valor)
    return [datetime.strptime(str(valor[k]), "%Y%m%d") for k in sorted(valor, key=lambda x: int(x))]


class MetadatosPASTIS:
    """Fold y fechas de cada parche. Lee metadata.geojson de PASTIS o el metadatos.json compacto
    que genera preparar_datos.py ({id: {"fold": f, "fechas": ["AAAAMMDD", ...]}})."""

    def __init__(self, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            gj = json.load(f)
        self.fold, self.fechas = {}, {}
        if "features" not in gj:                       # formato compacto
            for pid, v in gj.items():
                self.fold[int(pid)] = int(v["fold"])
                self.fechas[int(pid)] = [datetime.strptime(s, "%Y%m%d") for s in v["fechas"]]
            return
        for ft in gj["features"]:
            p = ft["properties"]
            pid = int(p["ID_PATCH"])
            self.fold[pid] = int(p["Fold"])
            self.fechas[pid] = _fechas_ordenadas(p["dates-S2"])

    def dias(self, pid):
        return np.array([(f - REFERENCE_DATE).days for f in self.fechas[pid]], dtype=np.int64)


def id_desde_nombre(nombre):
    """'S2_10001.npy' -> 10001 (None si el nombre no tiene ese formato)."""
    base = Path(str(nombre)).stem
    if base.startswith("S2_") and base[3:].isdigit():
        return int(base[3:])
    return None


# ------------------------------------------------------------------ TTA y ensamblado (idénticos al notebook)
def _transformar(x, k, flip, dims):
    x = torch.rot90(x, k, dims=dims)
    return torch.flip(x, dims=(dims[1],)) if flip else x


def _invertir(p, k, flip, dims):
    p = torch.flip(p, dims=(dims[1],)) if flip else p
    return torch.rot90(p, -k, dims=dims)


class Segmentador:
    """Carga TSViT una vez y segmenta parches PASTIS completos (128×128)."""

    def __init__(self, dispositivo=None, ruta_pesos=RUTA_PESOS):
        self.device = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ruta_pesos, map_location="cpu", weights_only=False)
        self.model = TSViTDias(ck["config"])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.epoca, self.miou_val = ck.get("epoca"), ck.get("miou_val")
        self.mean = np.asarray(ck["mean"], np.float32).reshape(1, 10, 1, 1)
        self.std = np.asarray(ck["std"], np.float32).reshape(1, 10, 1, 1)
        if self.device.type == "cuda":
            cc = torch.cuda.get_device_capability(self.device)
            self.amp = torch.bfloat16 if (torch.cuda.is_bf16_supported() and cc[0] >= 8) else torch.float16
        else:
            self.amp = None          # CPU: fp32

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp or torch.float32,
                              enabled=self.amp is not None)

    def preparar(self, x_bruto, dias):
        """Igual que cargar_parche(pid) sin rng: recorta a 60 fechas y normaliza."""
        x_bruto, dias = np.asarray(x_bruto), np.asarray(dias, dtype=np.int64)
        if x_bruto.ndim != 4 or x_bruto.shape[1:] != (10, 128, 128):
            raise ValueError(f"Se esperaba un arreglo (T, 10, 128, 128); llegó {x_bruto.shape}")
        if len(dias) != x_bruto.shape[0]:
            raise ValueError(f"El parche tiene {x_bruto.shape[0]} fechas pero hay {len(dias)} días")
        if len(dias) > MAX_SEQ_LEN:
            x_bruto, dias = x_bruto[:MAX_SEQ_LEN], dias[:MAX_SEQ_LEN]
        x = (x_bruto.astype(np.float32) - self.mean) / self.std
        return x, dias

    @torch.no_grad()
    def probabilidades(self, x_bruto, dias, tta=False):
        """Devuelve probabilidades (19, 128, 128) como numpy float32."""
        x, d = self.preparar(x_bruto, dias)
        xt = torch.zeros(MAX_SEQ_LEN, 128, 128, 10)
        xt[:x.shape[0]] = torch.from_numpy(x).permute(0, 2, 3, 1)
        dt = torch.zeros(MAX_SEQ_LEN, dtype=torch.long)
        dt[:len(d)] = torch.from_numpy(d)
        offs = OFFSETS_SOLAPE
        v = torch.stack([xt[:, r:r + VENTANA, c:c + VENTANA] for r in offs for c in offs]).to(self.device)
        dv = dt.to(self.device).expand(len(v), -1)
        prob = 0
        for k, f in (TRANSFORMACIONES if tta else [(0, False)]):
            vt = _transformar(v, k, f, (2, 3))
            out = []
            for j in range(0, len(vt), LOTE_INFERENCIA):
                with self._autocast():
                    out.append(self.model(vt[j:j + LOTE_INFERENCIA], dv[j:j + LOTE_INFERENCIA]).float().softmax(1))
            prob = prob + _invertir(torch.cat(out), k, f, (2, 3))
        prob = prob / (len(TRANSFORMACIONES) if tta else 1)
        return self._ensamblar(prob, offs).cpu().numpy()

    def _ensamblar(self, prob, offs):
        acum = torch.zeros(NUM_CLASSES, 128, 128, device=self.device)
        cuenta = torch.zeros(1, 128, 128, device=self.device)
        i = 0
        for r in offs:
            for c in offs:
                acum[:, r:r + VENTANA, c:c + VENTANA] += prob[i]
                cuenta[:, r:r + VENTANA, c:c + VENTANA] += 1
                i += 1
        return acum / cuenta.clamp(min=1)


# ------------------------------------------------------------------ resumen para el chatbot
def resumen_parche(prob, y=None):
    """Tabla por clase: píxeles, hectáreas, % del parche y confianza media de lo predicho.
    Si se da la etiqueta real y (128, 128), agrega hectáreas reales e IoU de ese parche
    (los píxeles Void/19 de la etiqueta se excluyen del IoU, como en las métricas del proyecto)."""
    pred = prob.argmax(0)
    conf = prob.max(0)
    filas = []
    for c in range(NUM_CLASSES):
        m = pred == c
        n = int(m.sum())
        fila = {"id": c, "clase": NOMBRES_CLASES[c], "pixeles_pred": n,
                "ha_pred": round(n * HA_POR_PIXEL, 2), "pct_parche": round(100 * n / pred.size, 2),
                "confianza_media": round(float(conf[m].mean()), 3) if n else None,
                "clase_debil": c in CLASES_DEBILES}
        if y is not None:
            valido = y != VOID_LABEL
            real = y == c
            n_real = int(real.sum())
            tp = int((m & real).sum())
            union = int(((m & valido) | real).sum())
            fila["ha_real"] = round(n_real * HA_POR_PIXEL, 2)
            fila["iou_parche"] = round(tp / union, 3) if union else None
        filas.append(fila)
    df = pd.DataFrame(filas)
    if y is not None:
        df.attrs["ha_sin_etiqueta"] = round(int((y == VOID_LABEL).sum()) * HA_POR_PIXEL, 2)
    df.attrs["confianza_global"] = round(float(conf.mean()), 3)
    return df
