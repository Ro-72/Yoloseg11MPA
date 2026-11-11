# Proyecto Jupyter — YOLO11 Segmentation para microplásticos

El proyecto contiene un único notebook autocontenido: `YOLO11_Seg_Microplasticos.ipynb`.

## Abrir

```powershell
Set-Location 'C:\Users\USER\Documents\Codex\2026-11-08\c-users-user-documents-codex-2026\outputs\yolo11_seg_jupyter'
.\run_jupyter.ps1
```

La instalación de PyTorch y Ultralytics puede ocupar bastante espacio. Actualmente conviene liberar espacio antes de crear el dataset completo.

## Controles iniciales

El notebook se entrega con:

```python
SMOKE_TEST = True
BUILD_DATASET = False
RUN_TRAINING = False
RUN_EVALUATION = False
```

Al abrirlo no generará tiles ni entrenará. Para construir primero una prueba pequeña, cambie solamente `BUILD_DATASET = True`. Para el flujo completo, utilice `SMOKE_TEST = False` y mantenga el entrenamiento desactivado hasta revisar el resumen del dataset.

## Decisiones metodológicas

- El split se realiza por `group_id` antes del tiling.
- Entrenamiento usa originales y aumentaciones; validación y prueba usan solo originales.
- Los tiles son de 640 × 640 con 128 píxeles de superposición.
- Se conservan todos los tiles positivos y una muestra determinista de negativos.
- Los polígonos se recortan al límite del tile y se exportan en formato YOLO Segmentation.
- Las aumentaciones geométricas internas de Ultralytics quedan desactivadas para no duplicar transformaciones.
- El entrenamiento está configurado con 1000 épocas como máximo, paciencia de 50 épocas y `mask_ratio: 1` para conservar las máscaras a resolución completa durante el entrenamiento.

La GPU AMD RX 6600 no funciona como dispositivo CUDA en PyTorch para Windows. El notebook permite una prueba en CPU, pero el entrenamiento completo se recomienda en un equipo con GPU NVIDIA o en un servicio como Colab/Kaggle.

