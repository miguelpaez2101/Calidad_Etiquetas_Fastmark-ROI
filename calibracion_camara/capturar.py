"""
Captura fotos del tablero de ajedrez para calibracion del lente ELP.
Controles: ESPACIO = guardar foto. Q = salir.
"""
import cv2
from pathlib import Path

CARPETA = Path(__file__).parent / "fotos"
CARPETA.mkdir(exist_ok=True)

# --- Abrir la camara ELP a resolucion real ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
cap.set(cv2.CAP_PROP_FPS, 12)

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara. Verifica el USB.")

# Parametros del tablero (para mostrar deteccion en tiempo real).
# Tablero chico: 9x7 cuadros de 13 mm -> 8x6 esquinas internas.
ESQUINAS_INTERNAS = (8, 6)

# Tamano del preview en pantalla. La deteccion se corre sobre esta version
# reducida — a 4000x3000 findChessboardCorners congela la GUI varios segundos.
# Al guardar (ESPACIO) se graba el frame original en resolucion completa.
PREVIEW_W, PREVIEW_H = 800, 600

contador = len(list(CARPETA.glob("*.jpg")))
print(f"Ya hay {contador} fotos guardadas en {CARPETA}")
print("ESPACIO = guardar foto  |  Q = salir")
print("Sugerencia: mueve e inclina el tablero para cada foto.\n")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Reduce antes de detectar para no congelar la GUI.
    preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
    gris_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    detectado, esquinas = cv2.findChessboardCorners(
        gris_preview, ESQUINAS_INTERNAS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
    )

    if detectado:
        cv2.drawChessboardCorners(preview, ESQUINAS_INTERNAS, esquinas, True)
        texto = f"TABLERO OK ({contador} fotos)"
        color = (0, 255, 0)
    else:
        texto = f"No detectado ({contador} fotos)"
        color = (0, 0, 255)

    cv2.putText(preview, texto, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Calibracion - ESPACIO guardar | Q salir", preview)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord(" "):
        if not detectado:
            print("  (no se detecto el tablero — mueve el tablero e intenta de nuevo)")
            continue
        ruta = CARPETA / f"cal_{contador:03d}.jpg"
        cv2.imwrite(str(ruta), frame)
        print(f"  Guardada: {ruta.name}")
        contador += 1
    elif tecla == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal fotos: {contador}")
