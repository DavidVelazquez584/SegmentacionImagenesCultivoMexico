"""Script base para preparar datos del proyecto."""

from segmentacion_cultivos_mx.config import PATHS
from segmentacion_cultivos_mx.data.preprocess import validate_data_directories


def main() -> None:
    validate_data_directories(PATHS.raw_data, PATHS.masks)
    print("Carpetas de datos validadas. Preparación pendiente de implementación.")


if __name__ == "__main__":
    main()
