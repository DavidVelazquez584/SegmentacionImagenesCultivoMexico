# ==============================================================================
# Capa de LLM intercambiable de Koré. El backend se elige con la variable LLM_BACKEND:
#   - "prueba": sin LLM; responde con la tabla del parche (para probar la interfaz).
#   - "ollama": modelo local servido por Ollama (modo presentación, sin internet).
#               Variables: OLLAMA_MODELO (p. ej. qwen2.5:3b), OLLAMA_URL (por defecto http://127.0.0.1:11434).
#   - "api":    Claude por la API de Anthropic (modo en línea, Hugging Face Spaces).
#   - "auto" (por defecto): Ollama si responde; si no, Claude si hay ANTHROPIC_API_KEY; si no, "prueba".
# Interfaz única: responder(prompt_sistema, historial) -> str, con historial en formato
# [{"role": "user"|"assistant", "content": str}, ...].
# ==============================================================================
import json
import os
import urllib.error
import urllib.request

TEMPERATURA = 0.2          # baja: queremos respuestas fieles a la tabla, no creativas
MAX_TURNOS = 12            # se envían como máximo los últimos 12 mensajes del historial


class BackendPrueba:
    nombre = "prueba (sin LLM)"

    def responder(self, prompt_sistema, historial):
        inicio = prompt_sistema.find("TABLA POR CLASE")
        fin = prompt_sistema.find("REGLAS")
        datos = prompt_sistema[inicio:fin].strip() if inicio >= 0 else "(sin datos)"
        return ("_Modo de prueba: no hay un LLM conectado._ "
                "Esta es la información que Koré le pasa al modelo de lenguaje:\n\n" + datos)


class BackendOllama:
    def __init__(self, modelo=None, url=None, timeout=180):
        self.modelo = modelo or os.environ.get("OLLAMA_MODELO", "ministral-3:3b")   # 10/10 en prueba_llm.py
        self.url = (url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout
        self.nombre = f"Ollama · {self.modelo}"
        self._sin_think = True     # pedir que no "piense" en voz alta (modelos con razonamiento)

    def _post(self, cuerpo):
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(cuerpo).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def responder(self, prompt_sistema, historial):
        cuerpo = {"model": self.modelo, "stream": False,
                  "messages": [{"role": "system", "content": prompt_sistema}] + historial[-MAX_TURNOS:],
                  "options": {"temperature": TEMPERATURA, "num_ctx": 8192}}
        if self._sin_think:
            cuerpo["think"] = False
        try:
            r = self._post(cuerpo)
        except urllib.error.HTTPError as e:
            if e.code == 400 and self._sin_think:      # el modelo no admite el parámetro "think"
                self._sin_think = False
                cuerpo.pop("think")
                r = self._post(cuerpo)
            else:
                raise RuntimeError(f"Ollama respondió {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
        except urllib.error.URLError:
            raise RuntimeError(f"no encontré el servidor de Ollama en {self.url}; ¿está abierto Ollama?")
        return r["message"]["content"].strip()


class BackendAnthropic:
    """Claude por la API de Anthropic (modo en línea). La clave se lee SOLO de la variable de entorno
    ANTHROPIC_API_KEY (en Hugging Face Spaces: un *secret*); nunca va en el código.
    Variables opcionales: ANTHROPIC_MODELO (por defecto claude-opus-5-5) y ANTHROPIC_ESFUERZO
    (por defecto "low": respuestas cortas de chat, menos tokens de razonamiento y menor costo)."""

    def __init__(self, modelo=None, esfuerzo=None):
        import anthropic                      # solo se importa si se usa este backend
        self._anthropic = anthropic
        self.modelo = modelo or os.environ.get("ANTHROPIC_MODELO", "claude-opus-5-5")
        self.esfuerzo = esfuerzo or os.environ.get("ANTHROPIC_ESFUERZO", "low")
        self.cliente = anthropic.Anthropic(timeout=60.0, max_retries=2)
        self.nombre = f"Claude · {self.modelo}"
        self.uso = {"entrada": 0, "cache_leido": 0, "salida": 0}

    def responder(self, prompt_sistema, historial):
        a = self._anthropic
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("falta configurar la clave ANTHROPIC_API_KEY (en el Space: Settings → Secrets)")
        try:
            r = self.cliente.beta.messages.create(
                model=self.modelo,
                max_tokens=8000,                    # incluye el razonamiento interno del modelo
                system=[{"type": "text", "text": prompt_sistema}],
                messages=historial[-MAX_TURNOS:],
                output_config={"effort": self.esfuerzo},
                cache_control={"type": "ephemeral"},   # reutiliza el prompt del parche entre turnos
                # Si el modelo declina por política, la API reintenta con un modelo de respaldo.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError:
            raise RuntimeError("la clave de la API no es válida o no está configurada (ANTHROPIC_API_KEY)")
        except a.RateLimitError:
            raise RuntimeError("se alcanzó el límite de uso de la API; intenta de nuevo en un minuto")
        except a.APIStatusError as e:
            raise RuntimeError(f"la API respondió con error {e.status_code}")
        except a.APIConnectionError:
            raise RuntimeError("no hay conexión con la API de Anthropic")
        u = r.usage                                   # tokens acumulados (para estimar costos)
        self.uso["entrada"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
        self.uso["cache_leido"] += u.cache_read_input_tokens or 0
        self.uso["salida"] += u.output_tokens
        if r.stop_reason == "refusal":
            return "No puedo responder esa pregunta. Pregúntame sobre los cultivos de este parche."
        texto = "".join(b.text for b in r.content if b.type == "text").strip()
        return texto or "No obtuve una respuesta; intenta reformular la pregunta."


def ollama_disponible(url=None):
    url = (url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=2):
            return True
    except Exception:
        return False


def obtener_backend(tipo=None):
    # "auto" (por defecto): Ollama si su servidor local responde; si no, Claude si hay clave de API
    # (caso del Space); si no, el modo de prueba.
    tipo = (tipo or os.environ.get("LLM_BACKEND", "auto")).lower()
    if tipo == "auto":
        if ollama_disponible():
            tipo = "ollama"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            tipo = "api"
        else:
            tipo = "prueba"
    if tipo == "prueba":
        return BackendPrueba()
    if tipo == "ollama":
        return BackendOllama()
    if tipo == "api":
        return BackendAnthropic()
    raise NotImplementedError(f"El backend '{tipo}' todavía no está implementado.")
