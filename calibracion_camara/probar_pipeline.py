"""
Valida el pipeline de vision end-to-end sobre una foto capturada con el
setup actual (300 mm, zoom/foco fijos).

Muestra un panel con tres imagenes lado a lado:
    1. Original — como sale de la camara.
    2. Undistortada — tras cv2.remap con calibracion.npz.
    3. Rectificada — tras rectificar_etiqueta (canvas canonico 3584x2048).

Uso:
    python3 probar_pipeline.py [ruta_foto]
Por defecto lee captura_temporal.jpg (lo que guardo capturar_uno.py).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))

from logica.cv_util import corregir_distorsion, rectificar_etiqueta  # noqa: E402

foto_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "captura_temporal.jpg"
img = cv2.imread(str(foto_arg))
if img is None:
    raise SystemExit(f"No se pudo leer {foto_arg}")

print(f"Procesando: {foto_arg}")
print(f"Original:    {img.shape[1]}x{img.shape[0]}")

undistortada = corregir_distorsion(img)
print(f"Undistort:   {undistortada.shape[1]}x{undistortada.shape[0]}")

# En el flujo real, Camara.capturar() ya devuelve la imagen undistortada
# (antes del recorte), y rectificar_etiqueta asume que el frame viene
# corregido.  Aquí replicamos ese contrato.
rectificada, razon = rectificar_etiqueta(undistortada)
if rectificada is not None:
    print(f"Rectificada: {rectificada.shape[1]}x{rectificada.shape[0]}  (OK)")
else:
    print(f"Rectificada: FALLO — razon='{razon}'")

# ── Layout del panel comparativo ────────────────────────────────────────────
ALTO = 400
SEP = 10
LABEL_H = 40


def redim(im: np.ndarray, alto: int = ALTO) -> np.ndarray:
    h, w = im.shape[:2]
    nuevo_ancho = int(w * alto / h)
    return cv2.resize(im, (nuevo_ancho, alto))


panel1 = redim(img)
panel2 = redim(undistortada)
if rectificada is not None:
    panel3 = redim(rectificada)
else:
    panel3 = np.zeros((ALTO, int(ALTO * 1.75), 3), dtype=np.uint8)
    cv2.putText(panel3, "FALLO", (40, ALTO // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(panel3, f"razon: {razon}", (40, ALTO // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

ancho_total = panel1.shape[1] + panel2.shape[1] + panel3.shape[1] + 2 * SEP
alto_total = ALTO + LABEL_H
canvas = np.full((alto_total, ancho_total, 3), 40, dtype=np.uint8)

def pegar(panel, x, titulo):
    canvas[LABEL_H:LABEL_H + ALTO, x:x + panel.shape[1]] = panel
    cv2.putText(canvas, titulo, (x + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


x = 0
pegar(panel1, x, "1. ORIGINAL")
x += panel1.shape[1] + SEP
pegar(panel2, x, "2. UNDISTORTADA")
x += panel2.shape[1] + SEP
pegar(panel3, x, "3. RECTIFICADA")

# Escalar si no cabe comoda en pantalla.
MAX_ANCHO = 1800
if canvas.shape[1] > MAX_ANCHO:
    escala = MAX_ANCHO / canvas.shape[1]
    canvas = cv2.resize(canvas, None, fx=escala, fy=escala)

cv2.imshow("Pipeline (tecla para salir)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
