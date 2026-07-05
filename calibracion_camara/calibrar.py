"""
Calibra el lente a partir de las fotos del tablero.
Guarda matriz intrinseca + coeficientes de distorsion en calibracion.npz.
"""
import cv2
import numpy as np
from pathlib import Path

# --- Parametros del tablero fisico ---
# Tablero chico: 9x7 cuadros de 13 mm -> 8x6 esquinas internas.
ESQUINAS_INTERNAS = (8, 6)   # (columnas, filas) — NO cuadros
LADO_CUADRO_MM    = 13.0

CARPETA_FOTOS = Path(__file__).parent / "fotos"
SALIDA        = Path(__file__).parent / "calibracion.npz"

# Coordenadas 3D teoricas de las esquinas del tablero (z=0 porque es plano).
# Estan en mm, escaladas por el tamano real del cuadro.
puntos_teoricos = np.zeros((ESQUINAS_INTERNAS[0] * ESQUINAS_INTERNAS[1], 3), np.float32)
puntos_teoricos[:, :2] = np.mgrid[0:ESQUINAS_INTERNAS[0],
                                  0:ESQUINAS_INTERNAS[1]].T.reshape(-1, 2)
puntos_teoricos *= LADO_CUADRO_MM

puntos_3d = []   # Coordenadas 3D del tablero (en mm) por cada foto valida.
puntos_2d = []   # Coordenadas 2D de esas esquinas en pixeles.
tamano_img = None
usadas, rechazadas = 0, 0

criterios_refinado = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

fotos = sorted(CARPETA_FOTOS.glob("*.jpg"))
if not fotos:
    raise SystemExit(f"No hay fotos en {CARPETA_FOTOS}. Corre capturar.py primero.")

print(f"Procesando {len(fotos)} fotos...\n")

for foto in fotos:
    img = cv2.imread(str(foto))
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if tamano_img is None:
        tamano_img = gris.shape[::-1]  # (w, h)

    ok, esquinas = cv2.findChessboardCorners(
        gris, ESQUINAS_INTERNAS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        print(f"  X {foto.name} — no se detecto el tablero")
        rechazadas += 1
        continue

    # Refina las esquinas a precision sub-pixel.
    esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterios_refinado)
    puntos_3d.append(puntos_teoricos)
    puntos_2d.append(esquinas)
    usadas += 1
    print(f"  OK {foto.name}")

print(f"\nFotos usadas: {usadas} | rechazadas: {rechazadas}")
if usadas < 10:
    raise SystemExit("Necesitas al menos 10 fotos buenas. Captura mas.")

print("\nCalibrando... (puede tardar 30-60 s en la Jetson)")
error, matriz_camara, coefs_distorsion, _, _ = cv2.calibrateCamera(
    puntos_3d, puntos_2d, tamano_img, None, None
)

print(f"\n--- Resultados ---")
print(f"Error de reproyeccion: {error:.3f} px")
print(f"  < 0.5 px  = muy bueno")
print(f"  0.5-1.0   = aceptable")
print(f"  > 1.0     = algo salio mal (tablero ondulado, pocas perspectivas)")
print(f"\nMatriz de camara (fx, fy, cx, cy):")
print(matriz_camara)
print(f"\nCoeficientes de distorsion (k1, k2, p1, p2, k3):")
print(coefs_distorsion.ravel())

np.savez(SALIDA,
         matriz_camara=matriz_camara,
         coefs_distorsion=coefs_distorsion,
         tamano_img=np.array(tamano_img),
         error_reproyeccion=error,
         n_fotos_usadas=usadas)
print(f"\nGuardado en {SALIDA}")
