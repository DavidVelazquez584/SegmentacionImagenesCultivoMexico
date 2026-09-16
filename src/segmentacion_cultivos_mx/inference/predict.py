"""Predicción de máscaras para imágenes nuevas."""

from pathlib import Path


def predict_image(image_path: Path, checkpoint_path: Path | None = None) -> Path:
    """Genera una máscara predicha para una imagen.

    Por ahora devuelve una ruta esperada de salida para dejar definido el contrato.
    """

    output_path = image_path.with_name(f"{image_path.stem}_mask_predicha.png")
    if checkpoint_path is not None:
        print(f"Checkpoint recibido: {checkpoint_path}")
    return output_path
