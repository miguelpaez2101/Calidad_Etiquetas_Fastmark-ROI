"""
Compara una foto antes/despues de enderezar con la calibracion obtenida.
Uso:
  python3 probar.py            -> usa fotos/cal_000.jpg
  python3 probar.py 9          -> usa fotos/cal_009.jpg
  python3 probar.py cal_015    -> usa fotos/cal_015.jpg
  python3 probar.py <ruta>     -> usa la ruta directa (absoluta o relativa)
"""
import cv2
import numpy as np
import sys
from pathlib import Path

AQUI = Path(__file__).parent
datos = np.load(AQUI / "calibracion.npz")
K = datos["matriz_camara"]
D = datos["coefs_distorsion"]


def resolver_ruta_foto(arg: str) -> Path:
    """Resuelve un argumento flexible a una ruta de foto real."""
    # Caso 1: numero puro ("9", "009")
    if arg.isdigit():
        return AQUI / "fotos" / f"cal_{int(arg):03d}.jpg"
    # Caso 2: nombre sin extension ("cal_015")
    p = Path(arg)
    if not p.suffix and not p.is_absolute() and len(p.parts) == 1:
        return AQUI / "fotos" / f"{arg}.jpg"
    # Caso 3: ruta directa — prueba como se pasa, luego relativa al script
    if p.exists():
        return p
    alternativa = AQUI.parent / arg
    if alternativa.exists():
        return alternativa
    return p  # devuelve original para que falle con mensaje claro


if len(sys.argv) > 1:
    ruta_foto = resolver_ruta_foto(sys.argv[1])
else:
    ruta_foto = AQUI / "fotos" / "cal_000.jpg"

img = cv2.imread(str(ruta_foto))
if img is None:
    raise SystemExit(f"No se pudo leer {ruta_foto}")
print(f"Procesando: {ruta_foto}")

# alpha=1 conserva todos los pixeles originales — los bordes negros curvos que
# aparezcan son la huella directa de la distorsion de barril siendo corregida.
# Para produccion conviene alpha=0 (recorta el borde util), pero para validar
# visualmente queremos ver el efecto completo.
h, w = img.shape[:2]
K_nueva, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
enderezada = cv2.undistort(img, K, D, None, K_nueva)

comparacion = np.hstack([img, enderezada])
# Ventana mas chica para que quepa comoda en la pantalla tactil 1920x1280.
ANCHO_PREVIEW = 1200
alto_preview = int(ANCHO_PREVIEW * comparacion.shape[0] / comparacion.shape[1])
preview = cv2.resize(comparacion, (ANCHO_PREVIEW, alto_preview))

medio = ANCHO_PREVIEW // 2
cv2.putText(preview, "ORIGINAL",   (20, 35),         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(preview, "ENDEREZADA", (medio + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Comparacion (presiona cualquier tecla para salir)", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
