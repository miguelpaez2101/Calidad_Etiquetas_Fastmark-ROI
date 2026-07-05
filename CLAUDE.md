# Prototipo Etiquetas ROI — Contexto para Claude

Variante **ligera** del sistema de inspección visual de etiquetas de
ROMAC / FastMark.  Fork de `../Prototipo_Etiquetas_Individuales`
(2026-07-04) reducido a un solo objetivo: **detectar manchones, oclusiones
y desvíos de color, rápido y sin GPU**.

## Qué es (y qué NO es)

Detecta comparando estadísticas HSV **por celda de 16 px** contra un banco
de etiquetas buenas — anclaje espacial: cada celda se compara solo contra
su propia historia en esa posición.  Eso atrapa el caso "cinta negra cubre
texto" que el PatchCore global del proyecto original dejaba pasar (los
parches oscuros matcheaban letras oscuras legítimas de otras zonas).

**Fuera de alcance a propósito** (usar el proyecto original para esto):
- Texto INCORRECTO con la misma paleta (lote/fecha mal impresos) → OCR.
- Defectos finos de textura (rayones delgados, blur, registro corrido)
  → PatchCore.
- Este fork NO tiene torch, paddle, GPU ni modelos de deep learning.

## Pipeline

```
Camara.capturar() → undistort → recorte global
  → rectificar_etiqueta          (segmentación a ESCALA_SEG=¼, kernels
                                  escalados, esquinas ×4, warp al canvas
                                  1024×576 — ~10 ms, paridad 0/18 vs
                                  el pipeline original a resolución completa)
  → DetectorROI.inspeccionar
      ├── verificar_presencia    (hist HSV global χ² + brillo — "sin_etiqueta")
      └── celdas 16 px           (z-score por celda vs μ/σ del banco;
                                  score = media top-4; "color" si > umbral)
```

Resultados: `OK` / `sin_luz` / `sin_etiqueta` / `color`.
Latencia total medida en laptop: **~30 ms por etiqueta** (~10 ms
rectificación + ~20 ms detector).  En Jetson Orin Nano se espera <150 ms —
validar.  El entrenamiento (estadísticas por celda + umbral leave-one-out)
tarda <1 s y corre en la propia Jetson sin liberar nada antes.

## Decisiones que NO hay que deshacer

- **Canvas 1024×576** (aspect 1.75 heredado): múltiplo de CELDA_PX=16
  (grilla 64×36).  El proyecto original usa 3584×2048 porque PatchCore y
  OCR lo necesitan; aquí sería puro costo.  El `.pkl` guarda canvas y
  celda y `cargar()` rechaza mismatches (reentrenar al cambiar).
- **Segmentación a ESCALA_SEG=0.25** (`cv_util.py`): misma lógica y
  umbrales relativos que el original, con kernels escalados para conservar
  el alcance físico (45→11, 25→7, 75→19, 15→5; FRANJA_LED 150→38).
  Error de re-escalado de esquinas ±4 px en el frame → ~±1 px en el canvas:
  irrelevante para estadísticas de color, sería inaceptable para OCR (que
  aquí no existe).  Validada con paridad 0/18 contra el original.
- **La trampa del trapecio sigue vigente**: `rectificar_etiqueta` ignora
  `rangos_hsv` y siempre segmenta con `RANGOS_HSV_NO_DOME` (V≥50).  Los
  rangos calibrados son para verificar contenido, nunca para segmentar el
  cartón (ver CLAUDE.md del proyecto original).
- **Umbral leave-one-out** (`detector_roi._scores_loo`): sin LOO los
  scores del banco son optimistas y el umbral queda bajo → falsos rechazos.
- **PISO_DIST_HIST=0.02** en la firma de presencia: un banco muy homogéneo
  colapsa `hist_dist_max` a ~0 y el tope 1.5× rechazaría capturas
  legítimas (visto en smoke test).  Una paleta ajena da χ² ≥0.3, el piso
  no debilita nada.

## Archivos por referencia

```
referencias/<Nombre>/
├── meta.json               — descripción + umbral override
├── calibracion_color.json  — paleta auto (k-means HSV, igual que el original)
├── plantilla_maestra.jpg   — captura usada para calibrar
├── modelo_roi.pkl          — μ/σ por celda + firma + umbral (schema v1)
└── buenas/buena_NNN.jpg    — banco rectificado (canvas 1024×576)
```

No hay `ocr_template.json` ni `patchcore.pkl` en esta variante.

## Relación con el proyecto original

- Compartimos: `hardware/camara.py`, la estructura de GUI, la calibración
  k-means (`calibracion_referencia.py`), el recorte global
  (`config/calibracion.json`) y la calibración de lente
  (`calibracion_camara/calibracion.npz`).
- **Arreglos en esos módulos deben portarse a mano entre ambos proyectos**
  (costo aceptado del fork; si esta variante se consolida, fusionar como
  "modo de inspección" en un solo repo).
- El detector nuevo (`logica/detector_roi.py`) generaliza la idea de
  `firma_color.py` del original (media±kσ por ROI manual) a una grilla
  fina automática.

## Hardware y entorno

Igual que el original (Jetson Orin Nano 8 GB, cámara ELP 48 MP @4000×3000,
domo con luz difusa, pantalla táctil) — pero SIN los patrones de memoria:
no hay OOM posible (footprint total <200 MB), no hay triple liberación,
no hay TensorRT ni Over-current como bloqueante de software.  En WSL/laptop
entra en modo simulación con `fotos_prueba/` (carpeta vacía en el fork —
copiar fotos del original si se necesita).

## Convenciones

Las mismas del proyecto original: comentarios y docstrings en español,
snake_case, señales Qt con `sig_`, dark theme inline, logging (no print).
