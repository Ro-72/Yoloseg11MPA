from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "YOLO11_Seg_Microplasticos.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# YOLO11 Segmentation para microplásticos

Notebook autocontenido para auditar las anotaciones Labelme, dividir por `group_id`, crear tiles de 640×640, recortar polígonos, exportar YOLO Segmentation y preparar el entrenamiento de `yolo11n-seg.pt`.

**Orden obligatorio:** auditoría → split por grupo → tiling → validación → entrenamiento. El entrenamiento está desactivado por defecto.
"""
    ),
    markdown("## 1. Controles de ejecución"),
    code(
        r"""
from pathlib import Path
import os

PROJECT_DIR = Path.cwd().resolve()
CONFIG_PATH = PROJECT_DIR / "configs" / "project.yaml"

# Solicitud interactiva de la carpeta que contiene images/ y annotations/.
# YOLO_DATASET_ROOT permite ejecutar pruebas automáticas sin bloquear el notebook.
dataset_root_value = os.environ.get("YOLO_DATASET_ROOT", "").strip().strip('"')
if not dataset_root_value:
    dataset_root_value = input(
        "Ingrese la carpeta que contiene images y annotations: "
    ).strip().strip('"')
if not dataset_root_value:
    raise ValueError("Debe ingresar una carpeta de datos.")
DATASET_ROOT = Path(dataset_root_value).expanduser().resolve()

# Valores seguros al abrir el notebook.
SMOKE_TEST = os.environ.get("YOLO_SMOKE_TEST", "1") == "1"
BUILD_DATASET = os.environ.get("YOLO_BUILD_DATASET", "0") == "1"
RUN_TRAINING = False
RUN_EVALUATION = False

print({
    "project": str(PROJECT_DIR),
    "dataset_root": str(DATASET_ROOT),
    "smoke_test": SMOKE_TEST,
    "build_dataset": BUILD_DATASET,
    "run_training": RUN_TRAINING,
    "run_evaluation": RUN_EVALUATION,
})
"""
    ),
    markdown("## 2. Importaciones y configuración"),
    code(
        r"""
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np
import yaml

config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
paths = config["paths"]
classes = {str(name): int(index) for name, index in config["classes"].items()}
id_to_class = {index: name for name, index in classes.items()}

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path

INPUT_IMAGES = DATASET_ROOT / "images"
INPUT_ANNOTATIONS = DATASET_ROOT / "annotations"
manifest_candidates = [
    DATASET_ROOT / "augmentation_manifest.csv",
    DATASET_ROOT / "manifests" / "augmentation_manifest.csv",
]
AUGMENTATION_MANIFEST = next((path for path in manifest_candidates if path.is_file()), None)
OUTPUT_ROOT = resolve_path(paths["smoke_dataset"] if SMOKE_TEST else paths["full_dataset"])

if not INPUT_IMAGES.is_dir():
    raise FileNotFoundError(f"No existe la carpeta de imágenes: {INPUT_IMAGES}")
if not INPUT_ANNOTATIONS.is_dir():
    raise FileNotFoundError(f"No existe la carpeta de anotaciones: {INPUT_ANNOTATIONS}")

print({
    "images": str(INPUT_IMAGES),
    "annotations": str(INPUT_ANNOTATIONS),
    "manifest": str(AUGMENTATION_MANIFEST) if AUGMENTATION_MANIFEST else None,
    "output": str(OUTPUT_ROOT),
})
"""
    ),
    markdown("## 3. Auditoría de imágenes, JSON y clases"),
    code(
        r"""
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def validate_labelme_pair(annotation_path, image_path):
    data = read_json(annotation_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer {image_path}")
    height, width = image.shape[:2]
    assert (int(data["imageWidth"]), int(data["imageHeight"])) == (width, height)
    for shape in data.get("shapes", []):
        assert shape.get("shape_type", "polygon") == "polygon"
        assert shape["label"] in classes
        assert len(shape["points"]) >= 3
    return data

annotation_paths = sorted(INPUT_ANNOTATIONS.glob("*.json"))
if not annotation_paths:
    raise FileNotFoundError(f"No se encontraron JSON de Labelme en {INPUT_ANNOTATIONS}")

manifest_rows = []
manifest_by_annotation = {}
if AUGMENTATION_MANIFEST:
    with AUGMENTATION_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))
    manifest_by_annotation = {
        Path(row["output_annotation"]).name: row
        for row in manifest_rows
        if row.get("output_annotation")
    }

groups = {}
missing_images = []
for annotation_path in annotation_paths:
    data = read_json(annotation_path)
    image_name = Path(data.get("imagePath", "")).name
    image_path = INPUT_IMAGES / image_name
    if not image_name or not image_path.is_file():
        candidates = sorted(INPUT_IMAGES.glob(f"{annotation_path.stem}.*"))
        image_path = candidates[0] if candidates else image_path
    if not image_path.is_file():
        missing_images.append(annotation_path.name)
        continue

    data = validate_labelme_pair(annotation_path, image_path)
    row = manifest_by_annotation.get(annotation_path.name, {})
    group_id = str(row.get("group_id") or image_path.stem)
    variant = str(row.get("variant") or "input")
    labels = {shape["label"] for shape in data.get("shapes", [])}
    group = groups.setdefault(group_id, {
        "group_id": group_id,
        "labels": set(),
        "polygon_count": 0,
        "samples": [],
    })
    group["labels"].update(labels)
    group["polygon_count"] += len(data.get("shapes", []))
    group["samples"].append({
        "image": image_path,
        "annotation": annotation_path,
        "variant": variant,
    })

if missing_images:
    raise FileNotFoundError(
        f"Hay {len(missing_images)} anotaciones sin imagen; primeras: {missing_images[:5]}"
    )
for group in groups.values():
    group["labels"] = sorted(group["labels"])

if manifest_rows and len(manifest_by_annotation) != len(annotation_paths):
    print(
        "ADVERTENCIA: el manifiesto no contiene todas las anotaciones; "
        "las faltantes se tratarán como grupos independientes."
    )
if not AUGMENTATION_MANIFEST:
    print(
        "ADVERTENCIA: no se encontró augmentation_manifest.csv. "
        "Cada imagen será considerada un grupo independiente."
    )

print({
    "groups": len(groups),
    "image_annotation_pairs": sum(len(item["samples"]) for item in groups.values()),
    "polygons": sum(item["polygon_count"] for item in groups.values()),
    "labels": Counter(label for item in groups.values() for label in item["labels"]),
})
"""
    ),
    markdown("## 4. División estratificada por `group_id`"),
    code(
        r"""
def deterministic_group_split(group_records, split_config):
    split_names = ("train", "val", "test")
    ratios = {name: float(split_config[name]) for name in split_names}
    total = len(group_records)
    targets = {
        "train": int(round(total * ratios["train"])),
        "val": int(round(total * ratios["val"])),
    }
    targets["test"] = total - targets["train"] - targets["val"]
    label_totals = Counter(label for item in group_records for label in item["labels"])
    label_targets = {
        split: {label: label_totals[label] * ratios[split] for label in classes}
        for split in split_names
    }
    rng = random.Random(int(split_config["seed"]))
    ordered = list(group_records)
    rng.shuffle(ordered)
    ordered.sort(
        key=lambda item: (
            "fragment" not in item["labels"],
            -item["polygon_count"],
        )
    )
    assigned = {name: [] for name in split_names}
    label_counts = {name: Counter() for name in split_names}
    for item in ordered:
        candidates = [name for name in split_names if len(assigned[name]) < targets[name]]
        def score(name):
            capacity = (targets[name] - len(assigned[name])) / max(targets[name], 1)
            deficit = sum(
                max(label_targets[name][label] - label_counts[name][label], 0)
                / max(label_totals[label], 1)
                for label in item["labels"]
            )
            return capacity + 2.0 * deficit
        selected = max(candidates, key=score)
        assigned[selected].append(item["group_id"])
        label_counts[selected].update(item["labels"])
    return assigned

splits = deterministic_group_split(list(groups.values()), config["split"])
assert set(splits["train"]).isdisjoint(splits["val"])
assert set(splits["train"]).isdisjoint(splits["test"])
assert set(splits["val"]).isdisjoint(splits["test"])
assert set().union(*map(set, splits.values())) == set(groups)

if SMOKE_TEST:
    limit = int(config["smoke_test"]["groups_per_split"])
    active_splits = {name: values[:limit] for name, values in splits.items()}
else:
    active_splits = splits

print("Split completo:", {name: len(values) for name, values in splits.items()})
print("Split activo:", {name: len(values) for name, values in active_splits.items()})
"""
    ),
    markdown("## 5. Geometría para recortar polígonos"),
    code(
        r"""
def polygon_area(points):
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    return abs(float(np.dot(array[:, 0], np.roll(array[:, 1], -1)) -
                     np.dot(array[:, 1], np.roll(array[:, 0], -1))) * 0.5)

def clip_edge(points, inside, intersect):
    if not points:
        return []
    output = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous, previous_inside = current, current_inside
    return output

def clip_polygon_to_rectangle(points, left, top, right, bottom):
    epsilon = 1e-12
    result = [[float(x), float(y)] for x, y in points]
    result = clip_edge(result, lambda p: p[0] >= left,
                       lambda a, b: [left, a[1] + (b[1]-a[1])*(left-a[0])/((b[0]-a[0]) or epsilon)])
    result = clip_edge(result, lambda p: p[0] <= right,
                       lambda a, b: [right, a[1] + (b[1]-a[1])*(right-a[0])/((b[0]-a[0]) or epsilon)])
    result = clip_edge(result, lambda p: p[1] >= top,
                       lambda a, b: [a[0] + (b[0]-a[0])*(top-a[1])/((b[1]-a[1]) or epsilon), top])
    result = clip_edge(result, lambda p: p[1] <= bottom,
                       lambda a, b: [a[0] + (b[0]-a[0])*(bottom-a[1])/((b[1]-a[1]) or epsilon), bottom])
    return result

def tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts
"""
    ),
    markdown("## 6. Construcción del plan de tiles 640×640"),
    code(
        r"""
def dataset_sources(active_split_map):
    records = []
    for split_name, group_ids in active_split_map.items():
        for group_id in group_ids:
            for sample in groups[group_id]["samples"]:
                records.append({
                    "split": split_name,
                    "group_id": group_id,
                    "kind": sample["variant"],
                    "image": sample["image"],
                    "annotation": sample["annotation"],
                })
    return records

def stable_seed(text, base_seed):
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")

def build_tile_plan(source_records, tiling_config):
    tile_size = int(tiling_config["tile_size"])
    overlap = int(tiling_config["overlap"])
    step = tile_size - overlap
    minimum_area = float(tiling_config["minimum_polygon_area_px"])
    minimum_fraction = float(tiling_config["minimum_retained_fraction"])
    plan = []
    for source in source_records:
        data = read_json(source["annotation"])
        width, height = int(data["imageWidth"]), int(data["imageHeight"])
        positives, negatives = [], []
        for y0 in tile_starts(height, tile_size, step):
            for x0 in tile_starts(width, tile_size, step):
                clipped_shapes = []
                for shape in data["shapes"]:
                    original_points = shape["points"]
                    original_area = polygon_area(original_points)
                    clipped = clip_polygon_to_rectangle(
                        original_points, x0, y0, x0 + tile_size, y0 + tile_size
                    )
                    local = [[x - x0, y - y0] for x, y in clipped]
                    clipped_area = polygon_area(local)
                    retained = clipped_area / max(original_area, 1e-9)
                    if len(local) >= 3 and clipped_area >= minimum_area and retained >= minimum_fraction:
                        clipped_shapes.append({"label": shape["label"], "points": local})
                record = {
                    **source,
                    "x": x0,
                    "y": y0,
                    "tile_size": tile_size,
                    "shapes": clipped_shapes,
                }
                (positives if clipped_shapes else negatives).append(record)
        ratio = (
            float(tiling_config["train_negative_ratio"])
            if source["split"] == "train"
            else float(tiling_config["evaluation_negative_ratio"])
        )
        negative_count = min(len(negatives), max(1, math.ceil(len(positives) * ratio)))
        rng = random.Random(stable_seed(source["image"].stem, config["split"]["seed"]))
        selected_negatives = rng.sample(negatives, negative_count) if negative_count else []
        plan.extend(positives + selected_negatives)
    return plan

sources = dataset_sources(active_splits)
tile_plan = build_tile_plan(sources, config["tiling"])
positive_tiles = sum(bool(record["shapes"]) for record in tile_plan)
estimated_bytes = len(tile_plan) * int(config["tiling"]["estimated_bytes_per_tile"])
free_bytes = shutil.disk_usage(OUTPUT_ROOT.parent).free

print({
    "source_images": len(sources),
    "selected_tiles": len(tile_plan),
    "positive_tiles": positive_tiles,
    "negative_tiles": len(tile_plan) - positive_tiles,
    "estimated_output_gb": round(estimated_bytes / 1024**3, 3),
    "free_disk_gb": round(free_bytes / 1024**3, 3),
})

if BUILD_DATASET and estimated_bytes > free_bytes * float(config["tiling"]["maximum_disk_fraction"]):
    raise RuntimeError("El dataset estimado supera el límite seguro de espacio disponible.")
"""
    ),
    markdown("## 7. Materialización y conversión a YOLO Segmentation"),
    code(
        r"""
def tile_basename(record):
    return f"{record['image'].stem}__x{record['x']:04d}_y{record['y']:04d}"

def yolo_lines(shapes, tile_size):
    lines = []
    for shape in shapes:
        class_id = classes[shape["label"]]
        coordinates = []
        for x, y in shape["points"]:
            coordinates.extend([
                f"{min(1.0, max(0.0, x / tile_size)):.6f}",
                f"{min(1.0, max(0.0, y / tile_size)):.6f}",
            ])
        lines.append(f"{class_id} " + " ".join(coordinates))
    return lines

def materialize_dataset(plan, output_root, tiling_config):
    output_root.mkdir(parents=True, exist_ok=True)
    for generated_name in ("images", "labels", "manifests", "previews"):
        generated_path = output_root / generated_name
        if generated_path.is_dir():
            shutil.rmtree(generated_path)
    grouped = defaultdict(list)
    for record in plan:
        grouped[record["image"]].append(record)
    manifest = []
    for image_path, records in grouped.items():
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo leer {image_path}")
        for record in records:
            split_name = record["split"]
            image_dir = output_root / "images" / split_name
            label_dir = output_root / "labels" / split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            name = tile_basename(record)
            size = record["tile_size"]
            crop = image[record["y"]:record["y"]+size, record["x"]:record["x"]+size]
            if crop.shape[:2] != (size, size):
                raise ValueError(f"Tile incompleto: {name}, forma={crop.shape}")
            image_output = image_dir / f"{name}.png"
            label_output = label_dir / f"{name}.txt"
            cv2.imwrite(
                str(image_output), crop,
                [cv2.IMWRITE_PNG_COMPRESSION, int(tiling_config["png_compression"])],
            )
            label_output.write_text("\n".join(yolo_lines(record["shapes"], size)) +
                                    ("\n" if record["shapes"] else ""), encoding="utf-8")
            manifest.append({
                "tile": name,
                "split": split_name,
                "group_id": record["group_id"],
                "source_image": image_path.name,
                "source_kind": record["kind"],
                "x": record["x"],
                "y": record["y"],
                "polygon_count": len(record["shapes"]),
            })
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "tiles_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in id_to_class.items()},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return manifest

tile_manifest = None
if BUILD_DATASET:
    tile_manifest = materialize_dataset(tile_plan, OUTPUT_ROOT, config["tiling"])
    print("Dataset escrito:", OUTPUT_ROOT, "tiles:", len(tile_manifest))
else:
    print("BUILD_DATASET=False: se calculó el plan, pero no se escribieron tiles.")
"""
    ),
    markdown("## 8. Validación y previsualizaciones"),
    code(
        r"""
def validate_yolo_dataset(output_root, manifest):
    failures = []
    groups_by_split = defaultdict(set)
    for row in manifest:
        groups_by_split[row["split"]].add(row["group_id"])
        image_path = output_root / "images" / row["split"] / f"{row['tile']}.png"
        label_path = output_root / "labels" / row["split"] / f"{row['tile']}.txt"
        if not image_path.is_file() or not label_path.is_file():
            failures.append(f"Falta par: {row['tile']}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (640, 640):
            failures.append(f"Imagen inválida: {image_path.name}")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) < 7 or (len(fields) - 1) % 2:
                failures.append(f"Etiqueta inválida {label_path.name}:{line_number}")
                continue
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if class_id not in id_to_class or not all(0.0 <= value <= 1.0 for value in values):
                failures.append(f"Valor inválido {label_path.name}:{line_number}")
    if groups_by_split["train"] & groups_by_split["val"]:
        failures.append("Fuga train-val")
    if groups_by_split["train"] & groups_by_split["test"]:
        failures.append("Fuga train-test")
    if groups_by_split["val"] & groups_by_split["test"]:
        failures.append("Fuga val-test")
    summary = {
        "tiles": len(manifest),
        "positive_tiles": sum(int(row["polygon_count"]) > 0 for row in manifest),
        "negative_tiles": sum(int(row["polygon_count"]) == 0 for row in manifest),
        "groups": {name: len(values) for name, values in groups_by_split.items()},
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    (output_root / "manifests" / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

def create_previews(output_root, manifest, limit=12):
    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in manifest if int(row["polygon_count"]) > 0][:limit]
    for row in selected:
        image_path = output_root / "images" / row["split"] / f"{row['tile']}.png"
        label_path = output_root / "labels" / row["split"] / f"{row['tile']}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            class_id = int(fields[0])
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
            points = np.rint(values.reshape(-1, 2) * 640).astype(np.int32)
            color = (40, 220, 40) if id_to_class[class_id] == "fiber" else (40, 150, 255)
            cv2.polylines(image, [points.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(preview_dir / f"{row['tile']}_overlay.png"), image)

if BUILD_DATASET:
    validation_summary = validate_yolo_dataset(OUTPUT_ROOT, tile_manifest)
    create_previews(OUTPUT_ROOT, tile_manifest)
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
    assert validation_summary["status"] == "PASS"
else:
    print("Validación omitida porque no se materializó el dataset.")
"""
    ),
    markdown("## 9. Entrenamiento opcional de YOLO11 Segmentation"),
    code(
        r"""
training_results = None
if RUN_TRAINING:
    if not (OUTPUT_ROOT / "data.yaml").is_file():
        raise FileNotFoundError("Primero construya y valide el dataset.")
    import torch
    from ultralytics import YOLO

    training = config["training"]
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Dispositivo de entrenamiento:", device)
    model = YOLO(training["model"])
    training_results = model.train(
        data=str(OUTPUT_ROOT / "data.yaml"),
        task="segment",
        imgsz=int(training["image_size"]),
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        mask_ratio=int(training["mask_ratio"]),
        batch=int(training["batch"]),
        workers=int(training["workers"]),
        device=device,
        seed=int(config["split"]["seed"]),
        project=str(PROJECT_DIR / "runs"),
        name="yolo11n_microplastic_seg",
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
else:
    print("RUN_TRAINING=False: no se descargaron pesos ni se inició entrenamiento.")
"""
    ),
    markdown("## 10. Evaluación opcional"),
    code(
        r"""
if RUN_EVALUATION:
    from ultralytics import YOLO
    best_weights = PROJECT_DIR / "runs" / "yolo11n_microplastic_seg" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)
    evaluation_model = YOLO(str(best_weights))
    metrics = evaluation_model.val(
        data=str(OUTPUT_ROOT / "data.yaml"),
        split="test",
        imgsz=640,
        workers=0,
    )
    print({
        "mask_mAP50_95": float(metrics.seg.map),
        "mask_mAP50": float(metrics.seg.map50),
        "box_mAP50_95": float(metrics.box.map),
        "box_mAP50": float(metrics.box.map50),
    })
else:
    print("RUN_EVALUATION=False: evaluación desactivada.")
"""
    ),
    markdown(
        r"""
## Resultado esperado

Cuando `BUILD_DATASET=True`, revise `validation_summary.json` y las imágenes de `previews` antes de activar el entrenamiento. Para el dataset completo cambie `SMOKE_TEST=False`; el notebook volverá a calcular el espacio necesario y se detendrá si no existe margen seguro.
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
