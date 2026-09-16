"""Script base para generar una máscara de segmentación sobre una imagen."""

from pathlib import Path

from segmentacion_cultivos_mx.inference.predict import predict_image


def main() -> None:
    example_path = Path("data/raw/imagen_ejemplo.png")
    output_path = predict_image(example_path)
    print(f"Ruta esperada de predicción: {output_path}")


if __name__ == "__main__":
    main()
