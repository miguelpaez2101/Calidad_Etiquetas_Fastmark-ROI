# Calidad_Etiquetas_Fastmark-ROI

Variante **ligera** del sistema de inspección de etiquetas ROMAC / FastMark
(repo hermano **Calidad_Etiquetas_Fastmark**), reducida a un objetivo:
**detectar manchones, oclusiones y desvíos de color — rápido y sin GPU**.

- Detector por **celdas de 16 px** (estadísticas HSV vs banco de buenas,
  umbral auto-calibrado leave-one-out) — numpy puro, CPU.
- Canvas reducido 1024×576 y segmentación a ¼ de escala:
  **~30 ms por etiqueta** en laptop (validado; en Jetson estimado <150 ms).
- Sin torch, sin PaddleOCR: desaparecen los patrones de OOM y las trampas
  de instalación aarch64 del proyecto completo.

**Fuera de alcance a propósito** (usar el proyecto completo): texto
incorrecto con la misma paleta (requiere OCR) y defectos finos de textura
(requiere PatchCore).  Detalles y decisiones de diseño en
[CLAUDE.md](CLAUDE.md).

## Puesta en marcha

1. Clonar y crear venv: `pip install -r requirements.txt`
   (en Jetson usar el OpenCV del sistema — el requirements ya excluye
   `opencv-python` en aarch64).
2. Configuración local (no versionada):
   ```
   cp config/calibracion.ejemplo.json config/calibracion.json
   ```
3. Calibración del lente versionada en
   `calibracion_camara/calibracion.npz` (SHA256
   `99021318f212fb45ca9321cc62e9e60d53526bd0608b2187fe946314fd1cbeb4`).
4. `python main.py` — sin cámara entra en simulación con `fotos_prueba/`
   (no versionada; copiar fotos del proyecto completo si se necesita).

## Flujo por referencia

1. "Calibrar referencia" → paleta automática k-means (sin OCR).
2. Capturar 15–30 buenas al banco.
3. "Entrenar modelo de color" → instantáneo (<1 s), en la propia Jetson.
4. RUN: `OK` / `sin_luz` / `sin_etiqueta` / `color` (las celdas
   defectuosas se dibujan en rojo sobre la vista rectificada).

## Qué NO está en el repo

`referencias/` (modelos y bancos — regenerables), `fotos_prueba/`,
imágenes de calibración, `logs/`, `config/calibracion.json` (config por
máquina — usar plantilla), `.claude/`.

## Flujo de trabajo

Editar y push desde la laptop (WSL); pull en la Jetson.  Los arreglos a
módulos compartidos con el proyecto completo (`cv_util`, `camara`, GUI)
se portan a mano entre ambos repos.
