"""Flujo base para evaluar modelos de segmentación."""

from pathlib import Path


def run_evaluation(checkpoint_path: Path | None = None) -> None:
    """Evalúa un modelo entrenado contra un conjunto de prueba."""

    message = "Evaluación pendiente de implementación."
    if checkpoint_path is not None:
        message = f"{message} Checkpoint recibido: {checkpoint_path}"
    print(message)
