# ==============================================================================
# Identidad visual de Koré: paleta de la interfaz y colores de las 19 clases de PASTIS.
# ==============================================================================
import gradio as gr

# ---------------------------------------------------------------- paleta de la interfaz
PALETA = {
    "primario": "#2E7D32",          # Verde cultivo
    "primario_oscuro": "#1B4D2E",   # Verde bosque (encabezado, hover)
    "secundario": "#1F5F8B",        # Azul satélite
    "acento": "#E0A526",            # Ámbar trigo (solo llamadas a la acción)
    "fondo": "#F6F4EC",             # Crema tierra
    "burbuja_bot": "#E4EFE1",       # Verde pálido
    "burbuja_usuario": "#DCE9F2",   # Azul pálido
    "texto": "#1E2A22",             # Carbón
    "texto_sec": "#5F6B61",         # Gris oliva
    "exito": "#43A047", "advertencia": "#F9A825", "error": "#C62828", "info": "#1F5F8B",
}
OSCURO = {"fondo": "#121A15", "superficie": "#1C2A21", "texto": "#E8EFE9",
          "primario": "#66BB6A", "acento": "#F2C14E"}

# ---------------------------------------------------------------- colores de clase
# La paleta propuesta trae 8 familias; PASTIS tiene 19 clases. Se conserva el tono de cada
# familia y se varía la luminosidad/saturación para que las clases de una misma familia sigan
# siendo distinguibles. Ningún color de clase repite el verde de la interfaz (#2E7D32).
COLORES_CLASES = {
    0: "#9E9E9E",   # Fondo (no agrícola)                 — familia Fondo
    1: "#2E9E5B",   # Pradera                             — Praderas / forraje
    2: "#E0B84A",   # Trigo blando de invierno            — Trigo / cereales
    3: "#8BC34A",   # Maíz
    4: "#B8862B",   # Cebada de invierno                  — cereales (oscuro)
    5: "#F2D100",   # Colza de invierno                   — Girasol / colza
    6: "#F3DE9A",   # Cebada de primavera                 — cereales (claro)
    7: "#F57F17",   # Girasol                             — Girasol / colza (naranja)
    8: "#7B3F8C",   # Vid                                 — Viñedo / frutales
    9: "#C2185B",   # Remolacha
    10: "#A67B5B",  # Triticale de invierno               — cereales (tierra)
    11: "#7A5C1E",  # Trigo duro de invierno              — cereales (muy oscuro)
    12: "#E57373",  # Frutas, hortalizas y flores
    13: "#5D4037",  # Papa
    14: "#00897B",  # Forraje leguminoso                  — Praderas / forraje (verde azulado)
    15: "#9E9D24",  # Soya
    16: "#B39DDB",  # Huerto                              — Viñedo / frutales (lavanda)
    17: "#FFE9A8",  # Cereal mixto                        — cereales (muy claro)
    18: "#3A8FB7",  # Sorgo                               — Agua / otros
    19: "#FFFFFF",  # Sin etiqueta (Void, solo en la etiqueta real)
}


def hex_a_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- tema de Gradio
def crear_tema():
    p = PALETA
    return gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
        body_background_fill=p["fondo"], body_background_fill_dark=OSCURO["fondo"],
        body_text_color=p["texto"], body_text_color_dark=OSCURO["texto"],
        body_text_color_subdued=p["texto_sec"],
        background_fill_primary="#FFFFFF", background_fill_primary_dark=OSCURO["superficie"],
        background_fill_secondary=p["burbuja_bot"], background_fill_secondary_dark="#16221B",
        block_background_fill="#FFFFFF", block_background_fill_dark=OSCURO["superficie"],
        block_border_color="#D9DED3", block_border_color_dark="#2C3B31",
        block_label_text_color=p["texto_sec"],
        block_title_text_color=p["primario_oscuro"], block_title_text_color_dark=OSCURO["primario"],
        border_color_primary="#D9DED3", border_color_primary_dark="#2C3B31",
        color_accent=p["primario"], color_accent_soft=p["burbuja_bot"], color_accent_soft_dark="#1F3326",
        link_text_color=p["secundario"], link_text_color_dark="#7FB8E0",
        button_primary_background_fill=p["acento"], button_primary_background_fill_dark=OSCURO["acento"],
        button_primary_background_fill_hover="#C98F17", button_primary_background_fill_hover_dark="#E0A526",
        button_primary_text_color=p["texto"], button_primary_text_color_dark="#121A15",
        button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark=OSCURO["superficie"],
        button_secondary_border_color=p["primario"], button_secondary_text_color=p["primario_oscuro"],
        button_secondary_text_color_dark=OSCURO["primario"],
        checkbox_background_color_selected=p["primario"],
        input_background_fill="#FFFFFF", input_background_fill_dark="#16221B",
    )


def html_login(logo_svg):
    """Contenido de la pantalla de inicio de sesión (parámetro auth_message de Gradio). Esa página no carga
    el CSS ni el <head> de la app, solo el tema; por eso trae sus propios estilos y fuentes."""
    p, o = PALETA, OSCURO
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600&display=swap');
@keyframes kl-aparecer {{ from {{ opacity: 0; transform: translateY(14px) scale(.98); }} to {{ opacity: 1; transform: none; }} }}
@keyframes kl-flotar {{ 0%, 100% {{ transform: translateY(0) rotate(-4deg); }} 50% {{ transform: translateY(-6px) rotate(4deg); }} }}
@keyframes kl-degradado {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
body, gradio-app, .gradio-container {{ min-height: 100vh;
  background: linear-gradient(120deg, {p['primario_oscuro']}, {p['primario']}, #2F6F5B, {p['secundario']}) !important;
  background-size: 260% 260% !important; animation: kl-degradado 20s ease infinite; }}
.gradio-container {{ max-width: none !important; padding: 0 !important; margin: 0 !important; }}
.gradio-container .main {{ padding: 0 !important; max-width: none !important; }}
.wrap.min-h-screen {{ display: grid !important; place-items: center; padding: 24px 16px !important;
  background: transparent !important; }}
.wrap.min-h-screen > .panel {{ width: min(420px, 100%); background: {p['fondo']} !important; border-radius: 22px !important;
  padding: 26px 26px 20px !important; box-shadow: 0 24px 60px #0B201480 !important; border: 1px solid #FFFFFF55 !important;
  animation: kl-aparecer .6s cubic-bezier(.2,.7,.2,1) both; font-family: 'Inter', system-ui, sans-serif; }}
.dark .wrap.min-h-screen > .panel {{ background: {o['superficie']} !important; border-color: #2C3B31 !important; }}
.wrap.min-h-screen > .panel > h2 {{ display: none !important; }}
.wrap.min-h-screen .block {{ border-radius: 12px !important; }}
.wrap.min-h-screen button.primary {{ border-radius: 12px !important; font-weight: 600; letter-spacing: .01em;
  transition: transform .15s ease, box-shadow .15s ease; }}
.wrap.min-h-screen button.primary:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px #E0A52666; }}
.kl-cab {{ text-align: center; margin-bottom: 6px; }}
.kl-cab svg {{ display: block; margin: 0 auto; width: 84px; height: 84px; filter: drop-shadow(0 8px 12px #6E142633); animation: kl-flotar 6s ease-in-out infinite;
  transform-origin: 50% 80%; }}
.kl-cab h1 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 2.6rem; line-height: 1; margin: 6px 0 4px;
  color: {p['primario_oscuro']}; }}
.dark .kl-cab h1 {{ color: {o['texto']}; }}
.kl-cab .lema {{ font-family: 'Fraunces', Georgia, serif; font-style: italic; color: #B3263E; font-size: 1rem; margin: 0 0 10px; }}
.dark .kl-cab .lema {{ color: {o['acento']}; }}
.kl-cab .txt {{ color: {p['texto_sec']}; font-size: .9rem; margin: 0 auto; max-width: 320px; line-height: 1.45; }}
.dark .kl-cab .txt {{ color: #A9B5AB; }}
.kl-sep {{ height: 1px; background: linear-gradient(90deg, transparent, #B3263E55, transparent); margin: 14px 0 4px; }}
@media (prefers-reduced-motion: reduce) {{ .wrap.min-h-screen, .wrap.min-h-screen > .panel, .kl-cab svg {{ animation: none !important; }} }}
</style>
<div class="kl-cab">{logo_svg}
  <h1>Koré</h1>
  <div class="lema">Como Perséfone en primavera, los campos vuelven a contarse.</div>
  <p class="txt">Chatbot de cultivos con TSViT y Sentinel-2 · Equipo 13, Tec de Monterrey.<br>
  Ingresa con el usuario y la contraseña que te compartió el equipo.</p>
  <div class="kl-sep"></div>
</div>"""


# Tipografía: Fraunces (serif con carácter clásico, para títulos y cifras) + Inter (texto de interfaz).
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700'
        '&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
        + """<script>
(function () {
  // Botón de tema: alterna la clase .dark de Gradio y recuerda la elección en este navegador.
  var CLAVE = 'kore-tema';
  function leer() { try { return localStorage.getItem(CLAVE); } catch (e) { return null; } }
  function guardar(v) { try { localStorage.setItem(CLAVE, v); } catch (e) {} }
  function nodos() { return [document.body].concat([].slice.call(document.querySelectorAll('gradio-app, .gradio-container'))); }
  function aplicar(oscuro) { nodos().forEach(function (el) { el.classList.toggle('dark', oscuro); }); }
  function esOscuro() { return nodos().some(function (el) { return el.classList.contains('dark'); }); }
  // Gradio pone su propia clase .dark (según el sistema) al montar la app: se respeta la elección guardada.
  new MutationObserver(function () {
    var v = leer(); if (!v) return; var quiero = v === 'oscuro';
    if (nodos().some(function (el) { return el.classList.contains('dark') !== quiero; })) aplicar(quiero);
  }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['class'] });
  document.addEventListener('click', function (ev) {
    if (!ev.target.closest || !ev.target.closest('#kore-tema')) return;
    var nuevo = !esOscuro(); aplicar(nuevo); guardar(nuevo ? 'oscuro' : 'claro');
  });
  function inicial() { var v = leer(); if (v) aplicar(v === 'oscuro'); }
  if (document.readyState !== 'loading') inicial(); else document.addEventListener('DOMContentLoaded', inicial);
  window.addEventListener('load', function () { inicial(); setTimeout(inicial, 600); });
})();
</script>""")

P = PALETA
CSS = f"""
:root {{
  --k-fondo: {P['fondo']}; --k-tarjeta: #FFFFFF; --k-borde: #DCE3D6; --k-texto: {P['texto']};
  --k-texto-sec: {P['texto_sec']}; --k-primario: {P['primario']}; --k-oscuro: {P['primario_oscuro']};
  --k-acento: {P['acento']}; --k-suave: {P['burbuja_bot']}; --k-sombra: 0 1px 2px #1B4D2E14, 0 8px 24px #1B4D2E12;
}}
.dark {{
  --k-fondo: {OSCURO['fondo']}; --k-tarjeta: {OSCURO['superficie']}; --k-borde: #2C3B31; --k-texto: {OSCURO['texto']};
  --k-texto-sec: #A9B5AB; --k-primario: {OSCURO['primario']}; --k-oscuro: #A5D6A7; --k-acento: {OSCURO['acento']};
  --k-suave: #1F3326; --k-sombra: 0 1px 2px #0006, 0 8px 24px #0005;
}}
.gradio-container {{ max-width: 1360px !important; margin: 0 auto !important; }}
.kore-serif, .kore-serif * {{ font-family: 'Fraunces', Georgia, serif !important; }}

/* ---------- animaciones ---------- */
@keyframes kore-aparecer {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes kore-flotar {{ 0%, 100% {{ transform: translateY(0) rotate(-3deg); }} 50% {{ transform: translateY(-4px) rotate(3deg); }} }}
@keyframes kore-degradado {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
@keyframes kore-fundido {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
.kore-anim {{ animation: kore-fundido .55s ease backwards; }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}

/* ---------- encabezado ---------- */
#kore-encabezado {{
  display: flex; align-items: center; gap: 18px; padding: 20px 26px; border-radius: 18px; color: #FFFFFF;
  background: linear-gradient(115deg, {P['primario_oscuro']}, {P['primario']}, #2F6F5B, {P['secundario']});
  background-size: 260% 260%; animation: kore-degradado 18s ease infinite; box-shadow: var(--k-sombra);
}}
#kore-encabezado .kore-logo {{ width: 74px; height: 74px; flex: none; filter: drop-shadow(0 6px 10px #0004);
  animation: kore-flotar 6s ease-in-out infinite; transform-origin: 50% 80%; }}
#kore-encabezado h1 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 2.6rem; line-height: 1;
  margin: 0; color: #FFFFFF; letter-spacing: .01em; }}
#kore-encabezado .lema {{ font-family: 'Fraunces', Georgia, serif; font-style: italic; color: #F3DE9A;
  font-size: 1.02rem; margin: 6px 0 2px; }}
#kore-encabezado p {{ color: #E4EFE1; margin: 0; font-size: .92rem; max-width: 760px; }}
#kore-encabezado .insignias {{ margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
#kore-encabezado .insignia {{ background: #FFFFFF1F; border: 1px solid #FFFFFF33; border-radius: 999px;
  padding: 5px 11px; font-size: .78rem; white-space: nowrap; backdrop-filter: blur(4px); }}
#kore-encabezado .insignia b {{ font-family: 'Fraunces', Georgia, serif; font-size: .95rem; }}
#kore-tema {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer; color: #FFFFFF;
  background: #FFFFFF26; border: 1px solid #FFFFFF55; border-radius: 999px; padding: 5px 12px;
  font: 600 .78rem 'Inter', sans-serif; transition: background .2s ease, transform .2s ease; }}
#kore-tema:hover {{ background: #FFFFFF40; transform: translateY(-1px); }}
#kore-tema:focus-visible {{ outline: 2px solid {P['acento']}; outline-offset: 2px; }}
#kore-tema .sol, #kore-tema .luna {{ font-size: 1rem; line-height: 1; }}
#kore-tema .sol {{ display: none; color: #F3DE9A; }}
.dark #kore-tema .sol {{ display: inline; }}
.dark #kore-tema .luna {{ display: none; }}
@media (max-width: 860px) {{
  #kore-encabezado {{ flex-wrap: wrap; padding: 16px; gap: 12px; }}
  #kore-encabezado .kore-logo {{ width: 56px; height: 56px; }}
  #kore-encabezado h1 {{ font-size: 2rem; }}
  #kore-encabezado .insignias {{ margin-left: 0; justify-content: flex-start; }}
}}

/* ---------- tarjetas y títulos de sección ---------- */
.kore-tarjeta {{ background: var(--k-tarjeta) !important; border: 1px solid var(--k-borde) !important;
  border-radius: 16px !important; box-shadow: var(--k-sombra); padding: 14px !important;
  animation: kore-fundido .55s ease backwards; }}
.kore-paso {{ display: flex; align-items: center; gap: 10px; margin: 2px 0 4px; }}
.kore-paso .num {{ width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; flex: none;
  background: var(--k-primario); color: #FFFFFF; font-weight: 600; font-size: .85rem; }}
.kore-paso h3 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.15rem; margin: 0; color: var(--k-oscuro); }}
#kore-boton {{ transition: transform .15s ease, box-shadow .15s ease; font-weight: 600; letter-spacing: .01em; }}
#kore-boton:hover {{ transform: translateY(-1px); box-shadow: 0 6px 16px #E0A52666; }}

/* ---------- tarjetas de resumen (KPIs) ---------- */
.kore-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(124px, 1fr)); gap: 10px; }}
.kore-kpi {{ background: var(--k-tarjeta); border: 1px solid var(--k-borde); border-radius: 14px; padding: 12px 14px;
  box-shadow: var(--k-sombra); animation: kore-aparecer .5s cubic-bezier(.2,.7,.2,1) both; position: relative; overflow: hidden; }}
.kore-kpi::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--k-primario); }}
.kore-kpi.acento::before {{ background: var(--k-acento); }}
.kore-kpi.azul::before {{ background: {P['secundario']}; }}
.kore-kpi.granada::before {{ background: #B3263E; }}
.kore-kpi .etq {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: var(--k-texto-sec); }}
.kore-kpi .val {{ font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 1.4rem; color: var(--k-texto);
  line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kore-kpi .sub {{ font-size: .8rem; color: var(--k-texto-sec); }}
.kore-kpi:nth-child(2) {{ animation-delay: .06s; }} .kore-kpi:nth-child(3) {{ animation-delay: .12s; }}
.kore-kpi:nth-child(4) {{ animation-delay: .18s; }} .kore-kpi:nth-child(5) {{ animation-delay: .24s; }}
.kore-vacio {{ border: 1.5px dashed var(--k-borde); border-radius: 14px; padding: 18px; text-align: center;
  color: var(--k-texto-sec); font-size: .92rem; }}
.kore-vacio b {{ font-family: 'Fraunces', Georgia, serif; color: var(--k-oscuro); font-size: 1.05rem; }}

/* ---------- mapas ---------- */
.kore-mapa img {{ border-radius: 10px; image-rendering: pixelated; transition: transform .25s ease; }}
.kore-mapa:hover img {{ transform: scale(1.015); }}

/* ---------- chat ---------- */
#kore-chat-col {{ position: sticky; top: 12px; align-self: flex-start; }}
#kore-chat .message-row {{ animation: kore-aparecer .35s ease both; }}
#kore-chat .bot.message, #kore-chat .message.bot {{ background: var(--k-suave) !important; border-color: transparent !important; }}
#kore-chat .user.message, #kore-chat .message.user {{ background: {P['burbuja_usuario']} !important; color: {P['texto']} !important;
  border-color: transparent !important; }}
.dark #kore-chat .user.message, .dark #kore-chat .message.user {{ background: #1F3A4F !important; color: {OSCURO['texto']} !important; }}

/* ---------- avisos, leyenda y pie ---------- */
.kore-aviso, .kore-error, .kore-ok {{ padding: 9px 12px; border-radius: 10px; font-size: .9rem; animation: kore-aparecer .4s ease both;
  color: {P['texto']} !important; }}
.kore-aviso {{ border-left: 4px solid {P['advertencia']}; background: #FFF6DB; }}
.kore-error {{ border-left: 4px solid {P['error']}; background: #FBE5E5; }}
.kore-ok {{ border-left: 4px solid {P['exito']}; background: #E7F4E8; }}
.kore-carga {{ padding: 9px 12px; border-radius: 10px; font-size: .9rem; display: flex; align-items: center; gap: 10px;
  border-left: 4px solid {P['secundario']}; background: #E3EEF6; color: {P['texto']} !important; animation: kore-fundido .25s ease backwards; }}
.dark .kore-carga {{ background: #1A2E3B; color: {OSCURO['texto']} !important; }}
.kore-girando {{ width: 16px; height: 16px; flex: none; border-radius: 50%; border: 2.5px solid #1F5F8B44;
  border-top-color: {P['secundario']}; animation: kore-girar .8s linear infinite; }}
@keyframes kore-girar {{ to {{ transform: rotate(360deg); }} }}

/* ---------- galería de ejemplos (chips) ---------- */
#kore-galeria .wrap {{ display: flex; flex-wrap: wrap; gap: 8px; }}
#kore-galeria label {{ border-radius: 999px !important; padding: 6px 12px !important; font-size: .86rem;
  border: 1px solid var(--k-borde) !important; background: var(--k-tarjeta) !important; cursor: pointer;
  transition: background .15s ease, border-color .15s ease; margin: 0 !important; }}
#kore-galeria label:hover {{ border-color: var(--k-primario) !important; }}
#kore-galeria label.selected, #kore-galeria label:has(input:checked) {{ background: var(--k-suave) !important;
  border-color: var(--k-primario) !important; color: var(--k-texto) !important; font-weight: 600; }}
#kore-galeria input[type=radio] {{ accent-color: {P['primario']}; }}

/* ---------- caja de subida ---------- */
#kore-subir [data-testid="upload-text"] {{ font-size: 0; line-height: 1.35; gap: 4px; }}
#kore-subir [data-testid="upload-text"] .or {{ display: none; }}
#kore-subir [data-testid="upload-icon"] {{ width: 22px; height: 22px; margin-bottom: 2px; }}
#kore-subir [data-testid="upload-text"]::after {{ content: "Arrastra aquí un S2_<ID>.npy\\Ao haz clic para buscarlo"; white-space: pre-line;
  font-size: .86rem; color: var(--k-texto-sec); text-align: center; }}
#kore-subir .upload-container {{ padding-top: 24px; }}
.kore-nota {{ margin-top: 8px; font-size: .8rem; color: var(--k-texto-sec); }}
#kore-acciones {{ justify-content: center; gap: 10px; }}
.kore-aviso *, .kore-error *, .kore-ok * {{ color: {P['texto']} !important; }}
.kore-leyenda {{ display: flex; flex-wrap: wrap; gap: 6px 8px; font-size: .82rem; }}
.kore-leyenda > span {{ background: var(--k-tarjeta); border: 1px solid var(--k-borde); border-radius: 999px; padding: 3px 10px 3px 6px; }}
.kore-leyenda span.chip {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
  border: 1px solid #00000033; vertical-align: -1px; margin-right: 6px; }}
.kore-pie {{ color: var(--k-texto-sec); font-size: .78rem; line-height: 1.5; }}
.kore-pie a {{ color: inherit; text-decoration: underline; }}
"""
