# ==============================================================================
# Koré — chatbot de cultivos sobre parches PASTIS segmentados en vivo con TSViT.
# Ejecutar:  python app.py          (LLM_BACKEND=auto|prueba|ollama|api)
# ==============================================================================
import os
from pathlib import Path

import gradio as gr
import numpy as np

import inferencia as inf
from contexto import construir_contexto
from estilo import COLORES_CLASES, CSS, HEAD, crear_tema, hex_a_rgb, html_login
from llm import obtener_backend

# ------------------------------------------------------------------ carga única
AQUI = Path(__file__).resolve().parent
# Datos compactos (preparar_datos.py): metadatos de los 2,433 parches, etiquetas del fold 5 y ejemplos.
DATOS = AQUI / "datos"
LOGO = AQUI / "assets" / "logo_kore.svg"
LOGO_SVG = LOGO.read_text(encoding="utf-8").replace("<svg ", '<svg class="kore-logo" ', 1)
META = inf.MetadatosPASTIS(DATOS / "metadatos.json")
ETIQUETAS_F5 = np.load(DATOS / "etiquetas_fold5.npz")
SEG = inf.Segmentador()
LLM = obtener_backend()
EN_GPU = SEG.device.type == "cuda"
LUT = np.array([hex_a_rgb(COLORES_CLASES[c]) for c in range(20)], dtype=np.uint8)
ESCALA = 3                     # 128 px -> 384 px en pantalla

# Parches de ejemplo del fold 5 (prueba), elegidos por variedad de cultivos.
EJEMPLOS = {
    40173: "Maíz, soya y girasol",
    10530: "Trigo blando dominante",
    20581: "Viñedo",
    40181: "Cebada de primavera y girasol",
    40305: "Colza de invierno",
    10101: "Trigo y remolacha",
    30024: "Soya, pradera y trigo",
    20392: "Trigo duro y hortalizas (clase débil)",
}
OPCIONES = [f"{pid} · {desc}" for pid, desc in EJEMPLOS.items()]
_cache = {}

KPIS_VACIO = ('<div class="kore-vacio"><b>Aún no hay un parche analizado</b><br>'
              'Elige un ejemplo del fold 5 (o sube un archivo) y pulsa <i>Analizar parche</i>.</div>')


# ------------------------------------------------------------------ utilidades de presentación
def a_color(mapa):
    return np.kron(LUT[mapa], np.ones((ESCALA, ESCALA, 1), dtype=np.uint8))


def fecha_despejada(x):
    """Índice de la fecha más útil para mostrar: sin nubes y con la vegetación más visible.

    Nubes: fracción de píxeles con azul (B2) > 2000. Entre las fechas con < 2 % de nubes se elige la de mayor
    NDVI medio; si ninguna está despejada, la menos nublada. Solo afecta la imagen mostrada, no al modelo."""
    x = np.asarray(x)
    azul = x[:, 0].astype(np.float32)
    nubes = (azul > 2000).mean(axis=(1, 2))
    con_datos = x.reshape(len(x), -1).any(axis=1)
    rojo, nir = x[:, 2].astype(np.float32), x[:, 6].astype(np.float32)
    ndvi = ((nir - rojo) / (nir + rojo + 1e-6)).mean(axis=(1, 2))
    candidatas = np.flatnonzero((nubes < 0.02) & con_datos)
    if len(candidatas) == 0:
        return int(np.argmin(np.where(con_datos, nubes, np.inf)))
    return int(candidatas[np.argmax(ndvi[candidatas])])


def rgb_verdadero(x, t):
    """Color verdadero (B4, B3, B2) de la fecha t, con estiramiento por percentiles 2–98 para que se vea bien."""
    b = np.asarray(x[t][[2, 1, 0]], dtype=np.float32).transpose(1, 2, 0)
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    img = (np.clip((b - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8)
    return np.kron(img, np.ones((ESCALA, ESCALA, 1), dtype=np.uint8))


def leyenda(clases):
    chips = "".join(f'<span><span class="chip" style="background:{COLORES_CLASES[c]}"></span>'
                    f'{"Sin etiqueta" if c == inf.VOID_LABEL else inf.NOMBRES_CLASES[c]}</span>'
                    for c in sorted(clases))
    return f'<div class="kore-leyenda">{chips}</div>'


def aviso(texto, tipo="aviso"):
    return f'<div class="kore-{tipo}">{texto}</div>'


def kpi(etiqueta, valor, sub, clase=""):
    return (f'<div class="kore-kpi {clase}"><div class="etq">{etiqueta}</div>'
            f'<div class="val">{valor}</div><div class="sub">{sub}</div></div>')


def tarjetas_resumen(df, pred, y, n_fechas, tta):
    agr = df[df["id"] > 0]
    dom = agr.loc[agr["ha_pred"].idxmax()]
    nombre = dom["clase"] if len(dom["clase"]) <= 16 else dom["clase"].split(" ")[0] + "…"
    tiles = [
        kpi("Cultivo dominante", nombre, f"{dom['ha_pred']:.2f} ha · {dom['pct_parche']:.1f} % del parche"),
        kpi("Superficie agrícola", f"{agr['ha_pred'].sum():.1f} ha", f"de {inf.HA_PARCHE:.2f} ha del parche", "acento"),
        kpi("Confianza media", f"{100 * df.attrs['confianza_global']:.1f} %", "probabilidad de la clase elegida", "azul"),
    ]
    if y is not None:
        valido = y != inf.VOID_LABEL
        exactitud = float((pred[valido] == y[valido]).mean())
        tiles.append(kpi("Exactitud vs. real", f"{100 * exactitud:.1f} %", "píxeles con etiqueta (fold 5)", "granada"))
    else:
        tiles.append(kpi("Etiqueta real", "Oculta", "parche usado en entrenamiento", "granada"))
    tiles.append(kpi("Clases detectadas", str(int(df["ha_pred"].gt(0).sum())),
                     f"{n_fechas} fechas · {'con' if tta else 'sin'} TTA"))
    return f'<div class="kore-kpis">{"".join(tiles)}</div>'


# ------------------------------------------------------------------ análisis de un parche
def _error(mensaje, tipo="error"):
    # Mismo orden que `salidas`: rgb, pred, real, leyenda, tabla, estado, chat, aviso, kpis
    return (None, None, None, "", None, None, [], aviso(mensaje, tipo), KPIS_VACIO)


def analizar(ejemplo, archivo, usar_tta):
    try:
        if archivo is not None:
            pid = inf.id_desde_nombre(Path(archivo).name)
            if pid is None:
                return _error("El archivo debe llamarse <b>S2_&lt;ID&gt;.npy</b>, como en PASTIS, "
                              "para poder recuperar sus fechas.")
            x = np.load(archivo, allow_pickle=False)
            origen = "archivo subido"
        elif ejemplo:
            pid = int(ejemplo.split(" ")[0])
            x = np.load(DATOS / "ejemplos" / f"S2_{pid}.npy")
            origen = "galería de ejemplos"
        else:
            return _error("Elige un parche de la galería o sube un archivo .npy.", "aviso")

        if pid not in META.fold:
            return _error(f"El ID {pid} no existe en PASTIS. Koré solo es válido para parches PASTIS "
                          "(Sentinel-2, Francia, 2018–2019), porque necesita sus fechas de adquisición.")
        dias, fold = META.dias(pid), META.fold[pid]
        if x.ndim != 4 or x.shape[1:] != (10, 128, 128) or x.shape[0] != len(dias):
            return _error(f"El arreglo tiene forma {x.shape}; para el parche {pid} se esperaba "
                          f"({len(dias)}, 10, 128, 128).")

        clave = (pid, bool(usar_tta), origen)
        if origen == "galería de ejemplos" and clave in _cache:
            prob = _cache[clave]
        else:
            prob = SEG.probabilidades(x, dias, tta=bool(usar_tta))
            if origen == "galería de ejemplos":
                _cache[clave] = prob
        pred = prob.argmax(0)

        # Regla anti-fuga: solo el fold 5 (prueba) se compara contra la etiqueta real.
        y = ETIQUETAS_F5[str(pid)].astype(np.int64) if (fold == 5 and str(pid) in ETIQUETAS_F5.files) else None
        df = inf.resumen_parche(prob, y)

        fechas = META.fechas[pid]
        info = {"id": pid, "fold": fold, "n_fechas": len(fechas), "fecha_ini": fechas[0].strftime("%d/%m/%Y"),
                "fecha_fin": fechas[-1].strftime("%d/%m/%Y"), "tiene_real": y is not None, "tta": bool(usar_tta)}
        sistema = construir_contexto(df, info)

        # Tabla visible
        visibles = df[(df["ha_pred"] > 0) | ((df["ha_real"] > 0) if y is not None else False)]
        visibles = visibles.sort_values("ha_pred", ascending=False)
        cols = {"clase": "Clase", "ha_pred": "Hectáreas (modelo)", "pct_parche": "% del parche",
                "confianza_media": "Confianza media"}
        if y is not None:
            cols.update({"ha_real": "Hectáreas (real)", "iou_parche": "IoU del parche"})
        tabla = visibles[list(cols)].rename(columns=cols)
        tabla.insert(0, "", ["⚠" if d else "" for d in visibles["clase_debil"]])

        clases_vis = set(np.unique(pred)) | (set(np.unique(y)) if y is not None else set())
        if y is not None:
            estado = aviso(f"Parche <b>{pid}</b> del fold 5 (prueba): el modelo nunca lo vio al entrenar, así que la "
                           f"comparación con la etiqueta real es una evaluación honesta · {origen}.", "ok")
        else:
            estado = aviso(f"Parche <b>{pid}</b> del fold {fold} ({'entrenamiento' if fold <= 3 else 'validación'}). "
                           "El modelo usó este parche para entrenar o para elegir su checkpoint, así que <b>no se "
                           "muestran métricas ni la etiqueta real</b>: compararlas sería fuga de datos.", "aviso")
        bienvenida = [{"role": "assistant", "content":
                       f"Listo, analicé el parche **{pid}** ({df['ha_pred'].gt(0).sum()} clases detectadas en "
                       f"{inf.HA_PARCHE:.2f} ha). Pregúntame, por ejemplo, *¿cuántas hectáreas son de maíz?*"}]
        t = fecha_despejada(x)
        img_rgb = gr.Image(value=rgb_verdadero(x, t),
                           label=f"Color verdadero · {fechas[t].strftime('%d/%m/%Y')}")
        img_real = (gr.Image(value=a_color(y), label="Etiqueta real (fold 5)") if y is not None else
                    gr.Image(value=None, label="Etiqueta real: oculta"))
        nota = (f'<div class="kore-nota">La foto en color verdadero usa la fecha más despejada de las {len(fechas)} '
                f'de la serie ({fechas[t].strftime("%d/%m/%Y")}); el modelo analiza todas.</div>')
        return (img_rgb, a_color(pred), img_real, leyenda(clases_vis) + nota,
                tabla, {"sistema": sistema, "pid": pid}, bienvenida, estado,
                tarjetas_resumen(df, pred, y, len(fechas), bool(usar_tta)))
    except Exception as e:                      # mostrar el error en la interfaz en lugar de romperla
        return _error(f"Error al analizar el parche: {e}")


# ------------------------------------------------------------------ chat
def conversar(mensaje, historial, estado):
    mensaje = (mensaje or "").strip()
    if not mensaje:
        return "", historial
    historial = list(historial or []) + [{"role": "user", "content": mensaje}]
    if not estado:
        respuesta = "Primero analiza un parche (elige uno de la galería o sube un archivo) y luego pregúntame."
    else:
        try:
            limpio = [{"role": m["role"], "content": m["content"]} for m in historial
                      if isinstance(m.get("content"), str)]
            respuesta = LLM.responder(estado["sistema"], limpio)
        except Exception as e:
            respuesta = f"No pude contactar al modelo de lenguaje ({LLM.nombre}): {e}"
    return "", historial + [{"role": "assistant", "content": respuesta}]


def paso(numero, titulo):
    return f'<div class="kore-paso"><span class="num">{numero}</span><h3>{titulo}</h3></div>'


# ------------------------------------------------------------------ interfaz
with gr.Blocks(title="Koré · cultivos desde el espacio") as demo:
    estado = gr.State(None)
    gr.HTML(f"""<header id="kore-encabezado" class="kore-anim">{LOGO_SVG}
      <div><h1>Koré</h1>
        <div class="lema">Como Perséfone en primavera, los campos vuelven a contarse.</div>
        <p>Segmenta en vivo una serie de tiempo Sentinel-2 con TSViT y conversa sobre sus cultivos.</p></div>
      <div class="insignias"><span class="insignia">mIoU <b>0.659</b></span>
        <span class="insignia">Macro F1 <b>0.781</b></span><span class="insignia">Accuracy <b>0.838</b></span>
        <span class="insignia">TSViT · PASTIS fold 5</span>
        <button id="kore-tema" type="button" title="Cambiar entre modo claro y oscuro"
                aria-label="Cambiar entre modo claro y oscuro"><span class="sol">☀︎</span><span class="luna">☾</span>
          <span class="txt">Tema</span></button></div></header>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, min_width=340):
            with gr.Column(elem_classes="kore-tarjeta"):
                gr.HTML(paso(1, "Elige un parche"))
                ejemplo = gr.Radio(OPCIONES, value=OPCIONES[0], elem_id="kore-galeria",
                                   label="Galería de ejemplos (fold 5: el modelo nunca los vio al entrenar)")
                with gr.Row(equal_height=True):
                    archivo = gr.File(label="O sube tu propio parche (opcional)", file_types=[".npy"],
                                      type="filepath", height=112, scale=1, min_width=220, elem_id="kore-subir")
                    with gr.Column(scale=1, min_width=220, elem_id="kore-acciones"):
                        tta = gr.Checkbox(value=EN_GPU, label="TTA · 8 transformaciones (más lento, +0.001 mIoU)")
                        boton = gr.Button("Analizar parche", variant="primary", elem_id="kore-boton")
                mensaje_estado = gr.HTML()
            html_kpis = gr.HTML(KPIS_VACIO)
            with gr.Column(elem_classes="kore-tarjeta"):
                gr.HTML(paso(2, "Mira el mapa"))
                with gr.Tabs():
                    with gr.Tab("Mapas"):
                        with gr.Row():
                            img_rgb = gr.Image(label="Color verdadero", interactive=False, height=260,
                                               elem_classes="kore-mapa")
                            img_pred = gr.Image(label="Mapa del modelo", interactive=False, height=260,
                                                elem_classes="kore-mapa")
                            img_real = gr.Image(label="Etiqueta real (fold 5)", interactive=False, height=260,
                                                elem_classes="kore-mapa")
                        html_leyenda = gr.HTML()
                    with gr.Tab("Hectáreas por clase"):
                        tabla = gr.Dataframe(label="1 píxel = 0.01 ha · parche = 163.84 ha · ⚠ = clase débil",
                                             interactive=False, wrap=True)
        with gr.Column(scale=5, min_width=320, elem_id="kore-chat-col"):
            with gr.Column(elem_classes="kore-tarjeta"):
                gr.HTML(paso(3, "Conversa con Koré"))
                chat = gr.Chatbot(show_label=False, height=520, elem_id="kore-chat",
                                  avatar_images=(None, str(LOGO)),
                                  placeholder="<b>Hola, soy Koré.</b><br>Analiza un parche y pregúntame por sus cultivos.")
                with gr.Row():
                    entrada = gr.Textbox(placeholder="Escribe tu pregunta…", show_label=False, scale=5, container=False)
                    enviar = gr.Button("Enviar", variant="primary", scale=1, min_width=90)
                gr.Examples(["¿Cuántas hectáreas son de maíz?", "¿Qué cultivo domina el parche?",
                             "¿Qué tan seguro está el modelo?", "¿Cuántas hectáreas de cereales hay?"],
                            inputs=entrada, label="Preguntas de ejemplo")

    gr.HTML(f"""<footer class="kore-pie">Modelo: TSViT entrenado desde cero (época {SEG.epoca}) ·
      dispositivo: {SEG.device.type.upper()} · LLM: {LLM.nombre}. Las hectáreas son estimaciones del modelo, no
      mediciones de campo.<br>Datos: PASTIS (Sainte Fare Garnot y Landrieu, ICCV 2021), licencia
      <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>; imágenes Copernicus
      Sentinel-2 procesadas por CNES/Theia; parcelas del RPG (IGN). Equipo 13 · Tec de Monterrey.</footer>""")

    salidas = [img_rgb, img_pred, img_real, html_leyenda, tabla, estado, chat, mensaje_estado, html_kpis]
    # Carga en tres pasos: aviso y botón deshabilitado al instante, análisis sin las capas de carga de Gradio
    # (que dejaban las imágenes a medio opacar) y botón habilitado de nuevo al terminar, aunque haya error.
    def _iniciar():
        return (aviso('<span class="kore-girando"></span>Analizando el parche… '
                      f'{"unos segundos con GPU" if EN_GPU else "puede tardar hasta un minuto en CPU"}.', "carga"),
                gr.Button(value="Analizando…", interactive=False))

    def _terminar():
        return gr.Button(value="Analizar parche", interactive=True)

    boton.click(_iniciar, None, [mensaje_estado, boton], queue=False, show_progress="hidden") \
         .then(analizar, [ejemplo, archivo, tta], salidas, show_progress="hidden") \
         .then(_terminar, None, boton, queue=False, show_progress="hidden")
    # Elegir un ejemplo de la galería descarta el archivo subido, para que quede claro cuál se analiza.
    ejemplo.input(lambda: None, None, archivo, queue=False, show_progress="hidden")
    entrada.submit(conversar, [entrada, chat, estado], [entrada, chat])
    enviar.click(conversar, [entrada, chat, estado], [entrada, chat])

if __name__ == "__main__":
    # Sin HOST/PORT, Gradio usa sus valores por defecto (127.0.0.1:7860 en local) o los que fija
    # Hugging Face Spaces mediante GRADIO_SERVER_NAME / GRADIO_SERVER_PORT.
    # KORE_COMPARTIR=1 crea un enlace público temporal (*.gradio.live) que apunta a esta computadora;
    # KORE_USUARIO y KORE_CLAVE (opcionales) le ponen usuario y contraseña.
    compartir = os.environ.get("KORE_COMPARTIR") == "1"
    acceso = ((os.environ["KORE_USUARIO"], os.environ["KORE_CLAVE"])
              if os.environ.get("KORE_USUARIO") and os.environ.get("KORE_CLAVE") else None)
    demo.launch(theme=crear_tema(), css=CSS, head=HEAD, favicon_path=str(LOGO),
                server_name=os.environ.get("HOST"),
                server_port=int(os.environ["PORT"]) if os.environ.get("PORT") else None, max_file_size="60mb",
                share=compartir, auth=acceso,
                auth_message=html_login(LOGO.read_text(encoding="utf-8")) if acceso else None)
