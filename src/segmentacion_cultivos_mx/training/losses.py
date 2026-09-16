"""Funciones de pérdida candidatas para segmentación."""


def available_losses() -> list[str]:
    """Lista pérdidas comunes para evaluar en fases posteriores."""

    return ["binary_cross_entropy", "dice_loss", "focal_loss"]
