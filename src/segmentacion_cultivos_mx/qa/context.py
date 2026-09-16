"""Construcción de contexto textual para un futuro componente de preguntas."""

from segmentacion_cultivos_mx.analysis.parcel_summary import CropSummary
from segmentacion_cultivos_mx.reporting.narrative import generate_crop_report


def build_qa_context(zone_name: str, crops: list[CropSummary], notes: list[str] | None = None) -> str:
    """Convierte resultados estructurados en contexto para un LLM."""

    report = generate_crop_report(zone_name=zone_name, crops=crops, notes=notes)
    guardrails = (
        "\n\nInstrucciones para responder preguntas: responde solo con base en los "
        "resultados anteriores. Si falta informacion, dilo explicitamente."
    )
    return report + guardrails
