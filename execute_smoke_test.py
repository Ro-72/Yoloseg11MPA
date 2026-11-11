from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "YOLO11_Seg_Microplasticos.ipynb"
OUTPUT_DIR = ROOT / "smoke_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["YOLO_SMOKE_TEST"] = "1"
os.environ["YOLO_BUILD_DATASET"] = "1"

with NOTEBOOK.open("r", encoding="utf-8") as file:
    notebook = nbformat.read(file, as_version=4)

client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()

executed_path = OUTPUT_DIR / "YOLO11_Seg_Microplasticos_smoke_executed.ipynb"
with executed_path.open("w", encoding="utf-8") as file:
    nbformat.write(notebook, file)

print(executed_path)
