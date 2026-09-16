"""Lógica principal de entrenamiento."""

from pathlib import Path


def run_training(config_path: Path | None = None) -> None:
    """Ejecuta el flujo de entrenamiento.

    Esta función queda como punto de entrada para integrar dataset, modelo,
    pérdidas, métricas y guardado de checkpoints.
    """

    message = "Entrenamiento pendiente de implementación."
    if config_path is not None:
        message = f"{message} Configuración recibida: {config_path}"
    print(message)
