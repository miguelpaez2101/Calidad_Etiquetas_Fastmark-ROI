"""
Captura una sola foto con la ELP a resolucion completa (4000x3000).
Uso:
    python3 capturar_uno.py [ruta_salida]
Si no se pasa ruta, guarda en calibracion_camara/captura_temporal.jpg.
Controles: ESPACIO = capturar y salir. Q = salir sin guardar.
"""
import sys
from pathlib import Path

import cv2

AQUI = Path(__file__).resolve().parent
SALIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "captura_temporal.jpg"

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
cap.set(cv2.CAP_PROP_FPS, 12)

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara. Verifica el USB.")

PREVIEW_W, PREVIEW_H = 800, 600
print("ESPACIO = capturar  |  Q = salir sin guardar")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
    cv2.putText(preview, "ESPACIO = capturar | Q = cancelar", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Captura oneshot", preview)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord(" "):
        cv2.imwrite(str(SALIDA), frame)
        print(f"Guardada: {SALIDA}  ({frame.shape[1]}x{frame.shape[0]})")
        break
    if tecla == ord("q"):
        print("Cancelado.")
        break

cap.release()
cv2.destroyAllWindows()
