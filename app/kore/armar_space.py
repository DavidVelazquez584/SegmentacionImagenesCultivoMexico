# Copia a ../kore_space/ solo lo que necesita el Space de Hugging Face (sin pruebas, cachés ni PASTIS).
# Uso: python armar_space.py
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DEST = AQUI.parent / "kore_space"
DEST.mkdir(exist_ok=True)
for viejo in DEST.iterdir():          # vacía la carpeta sin borrarla (puede estar abierta en otra ventana)
    shutil.rmtree(viejo) if viejo.is_dir() else viejo.unlink()

for f in ["app.py", "inferencia.py", "contexto.py", "estilo.py", "llm.py"]:
    shutil.copy(AQUI / f, DEST / f)
shutil.copy(AQUI / "space_README.md", DEST / "README.md")
shutil.copy(AQUI / "space_requirements.txt", DEST / "requirements.txt")
shutil.copytree(AQUI / "modelo", DEST / "modelo", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(AQUI / "datos", DEST / "datos")
shutil.copytree(AQUI / "assets", DEST / "assets")
(DEST / ".gitattributes").write_text("*.npy filter=lfs diff=lfs merge=lfs -text\n"
                                     "*.npz filter=lfs diff=lfs merge=lfs -text\n"
                                     "*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

total = 0
for f in sorted(DEST.rglob("*")):
    if f.is_file():
        total += f.stat().st_size
        print(f"{f.relative_to(DEST)}  ({f.stat().st_size / 1e6:.2f} MB)")
print(f"Total: {total / 1e6:.1f} MB en {DEST}")
