"""
cv_util - Utilidades comunes de OpenCV
======================================
- imread_safe/imwrite_safe: soportan rutas no-ASCII en Windows
  (cv2.imread/imwrite fallan silenciosamente con tildes/ñ porque usan
  la API ANSI internamente). Leer/escribir bytes con Python + imdecode
  /imencode evita el problema.
- segmentar_etiqueta: aísla la etiqueta del fondo oscuro del domo.
- rectificar_etiqueta: segmenta la etiqueta + recorta con bbox rotado +
  warpPerspective a un canvas canónico de tamaño fijo.  Cancela
  traslación y rotación: no importa dónde entra la etiqueta al domo,
  siempre se entrega "plana" en la misma coordenada.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

# ── Corrección de distorsión del lente ──────────────────────────────────────
# La calibración con tablero de ajedrez (calibracion_camara/) produce la matriz
# intrínseca y los coeficientes de distorsión.  Los aplicamos con cv2.remap
# sobre mapas precalculados (más rápido que cv2.undistort por frame) ANTES de
# cualquier otra operación espacial — segmentación, minAreaRect y warp asumen
# que las líneas rectas del mundo real salen rectas en la imagen.
#
# alpha=0 en getOptimalNewCameraMatrix: recorta los bordes curvos que deja la
# corrección del barril.  Evita bandas negras que podrían confundir al Otsu
# de segmentar_etiqueta (las tomaría como fondo del domo).

_RUTA_CALIBRACION = (
    Path(__file__).resolve().parent.parent / "calibracion_camara" / "calibracion.npz"
)

# Caché de mapas por tamaño de frame: clave (w, h) → (map1, map2) o None si
# no hay calibración disponible para ese tamaño.
_cache_mapas: dict[tuple[int, int], tuple[np.ndarray, np.ndarray] | None] = {}
_calibracion_advertida = False


def _cargar_calibracion() -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """
    Lee K, D y el tamaño de imagen con el que se calibró.
    Retorna None si el archivo no existe.
    """
    global _calibracion_advertida
    if not _RUTA_CALIBRACION.exists():
        if not _calibracion_advertida:
            logging.warning(
                "No se encontró %s — el pipeline correrá SIN corrección de "
                "distorsión del lente. Ejecuta calibracion_camara/calibrar.py.",
                _RUTA_CALIBRACION,
            )
            _calibracion_advertida = True
        return None
    datos = np.load(_RUTA_CALIBRACION)
    tamano = tuple(int(v) for v in datos["tamano_img"])  # (w, h)
    return datos["matriz_camara"], datos["coefs_distorsion"], tamano


def _obtener_mapas(w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Devuelve los mapas precalculados de remap para un frame (w, h).
    Los calcula la primera vez y los cachea.

    Retorna None si:
        - no hay calibración disponible, o
        - (w, h) NO coincide con el tamaño usado para calibrar.

    Este segundo caso es crítico: los coeficientes de distorsión asumen que
    (cx, cy) está en el centro óptico del sensor completo.  Aplicarlos sobre
    un frame recortado o de otra resolución produce una corrección
    geométricamente incorrecta.
    """
    clave = (w, h)
    if clave in _cache_mapas:
        return _cache_mapas[clave]

    cal = _cargar_calibracion()
    if cal is None:
        _cache_mapas[clave] = None
        return None

    K, D, (w_cal, h_cal) = cal
    if (w, h) != (w_cal, h_cal):
        logging.warning(
            "Undistort omitido: frame %dx%d no coincide con tamaño de "
            "calibración %dx%d.  Aplicar corrección sobre un frame recortado o "
            "de otra resolución produciría una corrección errónea.",
            w, h, w_cal, h_cal,
        )
        _cache_mapas[clave] = None
        return None

    K_nueva, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    # CV_16SC2: formato de mapa de enteros de 16 bits + tabla de fracción.
    # ~2x más rápido que CV_32FC1 en remap con pérdida de precisión despreciable
    # para corrección de distorsión.
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_nueva, (w, h), cv2.CV_16SC2)
    logging.info("Mapas de undistort calculados para frame %dx%d", w, h)
    _cache_mapas[clave] = (map1, map2)
    return map1, map2


def corregir_distorsion(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Cancela la distorsión radial/tangencial del lente usando la calibración
    guardada en calibracion_camara/calibracion.npz.

    Si no hay calibración disponible, devuelve la imagen sin modificar
    (con una advertencia en el log la primera vez).  Esto permite que el
    sistema siga funcionando en una Jetson recién instalada antes de
    calibrar, a costa de que el warp posterior estire bordes deformados.
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        return imagen_bgr
    h, w = imagen_bgr.shape[:2]
    mapas = _obtener_mapas(w, h)
    if mapas is None:
        return imagen_bgr
    map1, map2 = mapas
    return cv2.remap(imagen_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)


# ── Canvas canónico para la vista rectificada ───────────────────────────────
# VARIANTE ROI: canvas reducido 1024×576 (aspect 1.75, igual que el proyecto
# original para heredar su geometría).  El detector por celdas de color no
# necesita la resolución 3584×2048 del proyecto completo (esa existía por
# los tiles de 512 px de PatchCore y la densidad de texto para OCR): las
# estadísticas HSV por celda son estables a esta escala y todo el pipeline
# (warp, conversión HSV, features) se abarata ~12×.  Ambos lados son
# múltiplos de CELDA_PX=16 del detector (1024=64·16, 576=36·16 → grilla
# 64×36).  Si se cambia, mantener múltiplos de 16 y reentrenar (el .pkl
# guarda el canvas y rechaza mismatches).
CANONICAL_W = 1024
CANONICAL_H = 576

# Escala a la que corre la SEGMENTACIÓN dentro de rectificar_etiqueta.
# Segmentar sobre el frame completo (4000×3000 recortado) es lo más caro del
# pipeline CPU (kernels de morfología 45/75 px).  A ¼ de escala la morfología
# cuesta ~16× menos y las esquinas detectadas se re-escalan al frame completo
# para el warp: el error de re-escalado es ±ESCALA⁻¹ px en el frame (±4 px),
# que proyectado al canvas de 1024 px queda en ~±1 px — irrelevante para
# estadísticas de color por celdas de 16 px (sería inaceptable para OCR,
# pero esta variante no tiene OCR).
ESCALA_SEG = 0.25


# ── Segmentación de etiqueta sobre fondo oscuro ──────────────────────────────

# Área mínima (fracción del frame) para aceptar una segmentación como válida.
# Por debajo se asume que no hay etiqueta o que la iluminación falló.
_AREA_MIN_ETIQUETA_FRAC = 0.10

# ── Rangos HSV: fallback Lannate y soporte por-referencia ──────────────────
# Históricamente segmentábamos buscando tres regiones de color del Lannate:
#   - banner verde (arriba): H en [35, 90], S > 30, V > 30.
#   - cuerpo blanco-grisáceo (centro): V > 130, S < 70.
#   - franja roja (abajo): H en [0, 15] ∪ [165, 180], S > 40, V > 30.
# La unión de las tres capta el label completo aunque Otsu sobre V
# fragmente el banner verde topográfico (líneas oscuras internas con V cerca
# del valle de Otsu).  Por qué color y no Otsu:
#   - Otsu elige un umbral global por imagen (110-120 V típico) y en capturas
#     con el banner menos saturado fragmenta el banner en 5-10 blobs
#     pequeños → el cuerpo queda como blob mayor y el banner se descarta →
#     la rectificada sale sin banner.
#   - Las sombras del label sobre la pared interior del domo son acromáticas
#     (saturación muy baja, V medio).  En "blanco" clasifican si V > 130,
#     pero el OPEN agresivo (45×45) las elimina por ser finas.
#
# Para soportar referencias con paleta distinta (Harvanta con banner azul,
# etc.) los rangos viven ahora en `referencias/<ref>/calibracion_color.json`
# y los callers se los pasan a `segmentar_etiqueta` como `rangos_hsv`.  Estos
# valores quedan como FALLBACK cuando no hay calibración disponible (modo
# compatibilidad con Lannate, o calibración corrupta).
_RANGOS_HSV_FALLBACK_LANNATE: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array((35,  30,  30), dtype=np.uint8),  # banner verde lo
     np.array((90, 255, 255), dtype=np.uint8)),
    (np.array((0,    0, 130), dtype=np.uint8),  # cuerpo blanco lo
     np.array((180, 70, 255), dtype=np.uint8)),
    (np.array((0,   40,  30), dtype=np.uint8),  # rojo bajo
     np.array((15, 255, 255), dtype=np.uint8)),
    (np.array((165, 40,  30), dtype=np.uint8),  # rojo alto (wraparound)
     np.array((180, 255, 255), dtype=np.uint8)),
]

# Rango "no-dome" universal: captura cualquier píxel que NO sea fondo
# (paredes negras del domo, sombras profundas).  Útil como bootstrap para
# detectar el perímetro de una etiqueta nueva ANTES de tener su calibración
# personalizada.
#
# Sólo V ≥ 50.  No usamos OR con S ≥ algo porque en zonas oscuras (V<10)
# el cálculo de S = (max-min)/max queda inestable y ruido de cuantización
# da S>30 en píxeles que claramente son fondo — eso volvía la máscara
# bootstrap a 90 %+ del frame y hacía que el contorno tocara los bordes.
# Con sólo V ≥ 50, los huecos oscuros internos del label (letras negras,
# iconos PELIGRO, regiones rojas oscuras) los rellena el close grande del
# pipeline tras seleccionar el componente conexo mayor.
RANGOS_HSV_NO_DOME: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array((0,   0, 50), dtype=np.uint8),
     np.array((180, 255, 255), dtype=np.uint8)),
]

# Kernels de morfología.
# - Apertura 45×45 (ELLIPSE): destruye reflejos/sombras finas y puentes de
#   halo entre el label y el domo.  El label es masivo (>4 M px²) y sobrevive
#   sin perder bordes; ruidos con ancho <45 px desaparecen.
# - Cierre "pre-CC" 25×25 × 2 iter (alcance ~50 px): une las tres regiones de
#   color del label (verde/blanco/rojo) cuando la divisoria de bajo contraste
#   deja gap entre ellas.  50 px es suficiente para los gaps observados
#   (15-50 px) y no alcanza reflejos laterales (que quedan a ≥60 px tras
#   el OPEN 45).  Se aplica ANTES del connectedComponents para que el label
#   quede como un único blob grande.
# - Cierre final 75×75 × 3 iter (alcance 225 px): rellena huecos internos
#   del banner topográfico (líneas oscuras de ~200 px).  Se aplica DESPUÉS
#   de elegir el componente principal para que no conecte contaminación
#   externa.
# - Pulido 15×15: suaviza el contorno final antes de encontrar vértices.
# Tamaños BASE de los kernels, calibrados sobre el frame a resolución
# completa (4000×3000 recortado).  Cuando la segmentación corre a escala
# reducida (ESCALA_SEG), los kernels se escalan proporcionalmente para
# conservar el mismo alcance FÍSICO en mm — ver _kernels_para_escala.
_KERNEL_BASE_APERTURA   = 45
_KERNEL_BASE_PRE_CIERRE = 25
_KERNEL_BASE_CIERRE     = 75
_KERNEL_BASE_PULIDO     = 15


def _impar(n: float) -> int:
    """Redondea a entero impar ≥ 3 (los kernels de morfología deben ser impares)."""
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


_cache_kernels: dict[float, dict] = {}


def _kernels_para_escala(escala: float) -> dict:
    """
    Kernels de morfología escalados.  A escala 1.0 reproduce exactamente los
    del proyecto original (45/25/75/15); a 0.25 → 11/7/19/5, que conservan
    el mismo alcance físico sobre la imagen reducida.
    """
    if escala in _cache_kernels:
        return _cache_kernels[escala]
    k = {
        "apertura":   cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_impar(_KERNEL_BASE_APERTURA * escala),) * 2),
        "pre_cierre": cv2.getStructuringElement(
            cv2.MORPH_RECT, (_impar(_KERNEL_BASE_PRE_CIERRE * escala),) * 2),
        "cierre":     cv2.getStructuringElement(
            cv2.MORPH_RECT, (_impar(_KERNEL_BASE_CIERRE * escala),) * 2),
        "pulido":     cv2.getStructuringElement(
            cv2.MORPH_RECT, (_impar(_KERNEL_BASE_PULIDO * escala),) * 2),
    }
    _cache_kernels[escala] = k
    return k

# Altura de la franja de borde (arriba/abajo) que inspeccionamos en busca de
# tiras LED del domo.  El label útil debe estar a ≥ FRANJA_LED_PX del borde
# superior/inferior del frame — el recorte de la cámara debería garantizarlo.
FRANJA_LED_PX = 150
# Fracción de brillo (ratio píxeles blancos / área de la franja) por encima
# de la cual consideramos que la franja contiene tira LED y hay que limpiarla.
# 20 % evita falsos positivos con reflejos débiles o bordes del label asomando.
UMBRAL_BRILLO_FRANJA_LED = 0.20


def _limpiar_franjas_led(mask_binaria: np.ndarray,
                         franja_px: int = FRANJA_LED_PX) -> np.ndarray:
    """
    Limpia las franjas superior/inferior del frame si tienen evidencia de tiras
    LED (mucho brillo concentrado arriba/abajo).

    Retorna la máscara modificada in-place: las franjas detectadas como "con
    LED" se ponen a 0, el resto se deja igual.  Si las franjas no superan
    el umbral (ej. iluminación tenue o LEDs no visibles), no modifica nada.

    Se aplica DESPUÉS de Otsu + apertura y ANTES del cierre agresivo — así
    el cierre no puede fusionar las LEDs con el cuerpo de la etiqueta.
    """
    H, W = mask_binaria.shape[:2]
    if H <= 2 * franja_px:
        return mask_binaria

    area_franja = franja_px * W
    umbral_px   = UMBRAL_BRILLO_FRANJA_LED * area_franja

    if np.count_nonzero(mask_binaria[:franja_px, :]) > umbral_px:
        mask_binaria[:franja_px, :] = 0
    if np.count_nonzero(mask_binaria[H - franja_px:, :]) > umbral_px:
        mask_binaria[H - franja_px:, :] = 0

    return mask_binaria


def imread_safe(ruta: str) -> np.ndarray | None:
    """
    Equivalente a cv2.imread() pero soporta rutas con caracteres no-ASCII.
    Retorna None si el archivo no existe o no se puede decodificar.
    """
    try:
        with open(ruta, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except (OSError, IOError):
        return None


def imwrite_safe(ruta: str, imagen: np.ndarray) -> bool:
    """
    Equivalente a cv2.imwrite() pero soporta rutas con caracteres no-ASCII.
    Detecta el formato por la extensión del archivo (.jpg, .png, etc.).
    Retorna True si se guardó correctamente, False si falló.
    """
    import os
    ext = os.path.splitext(ruta)[1].lower() or ".jpg"
    ret, buf = cv2.imencode(ext, imagen)
    if not ret:
        return False
    try:
        with open(ruta, "wb") as f:
            f.write(buf.tobytes())
        return True
    except (OSError, IOError):
        return False


def segmentar_etiqueta(
    imagen_bgr: np.ndarray,
    rangos_hsv: list[tuple[np.ndarray, np.ndarray]] | None = None,
    escala: float = 1.0,
) -> np.ndarray | None:
    """
    Aísla la etiqueta del fondo oscuro del domo usando sus regiones de color.

    Parámetros:
        imagen_bgr: frame de la cámara (BGR, undistortado).
        rangos_hsv: lista de pares (lo, hi) para `cv2.inRange`.  Cada par
            describe una región de color del label (banner, cuerpo, franja).
            Si es None, se usa el fallback con la paleta del Lannate — útil
            para mantener compatibilidad con código viejo que aún no propaga
            la calibración por-referencia.

    Pipeline:
        1. Máscara HSV = unión de los rangos definidos por la referencia
           (verde ∪ blanco ∪ rojo en Lannate; otra combinación en cada
           referencia calibrada).
        2. Apertura 45×45: elimina reflejos/sombras con ancho < 45 px
           (halos del label sobre el domo, ruido lateral).  El label
           sobrevive porque es masivo.
        3. Limpieza de franjas LED (elimina tiras LED del domo visibles
           en el borde superior/inferior del frame, que el umbral "blanco"
           atrapa por tener V alto y S bajo).
        4. Cierre pre-CC 25×25 × 2 iter: une las regiones de color del label
           cuando las divisorias de bajo contraste las dejan como blobs
           separados por 15-50 px.  Alcance ~50 px.
        5. Componentes conexas → blob mayor = label completo.
        6. Cierre grande 75×75 × 3 iter: rellena huecos topográficos
           internos del banner (~200 px de gap).  Opera SÓLO sobre el blob
           mayor.
        7. Pulido 15×15.

    Retorna:
        - np.ndarray uint8 (H, W) con 255 dentro de la etiqueta, 0 fuera.
        - None si la segmentación falla (área < _AREA_MIN_ETIQUETA_FRAC del frame).
          Los llamadores deben tratar ese caso como "sin_etiqueta".

    Por qué por color y no por Otsu sobre V:
        Otsu elige un umbral global por imagen (~110-120 en V).  En capturas
        donde el banner tiene menos saturación de la habitual, ese umbral
        corta el banner en muchos fragmentos pequeños → ninguno supera el
        área del cuerpo → el blob mayor es solo cuerpo y el banner desaparece
        de la máscara → la rectificada sale sin banner.  Calculando máscaras
        separadas por color y uniéndolas, el banner forma un blob consistente
        en todas las capturas.
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        return None

    # Segmentación a escala reducida (VARIANTE ROI): la morfología domina el
    # costo y escala ~cuadráticamente con la resolución.  A escala<1 el frame
    # se reduce, los kernels se escalan para conservar el alcance físico, y
    # la MÁSCARA DEVUELTA queda en el espacio reducido — el caller
    # (rectificar_etiqueta) re-escala las esquinas al frame original.
    if escala != 1.0:
        imagen_bgr = cv2.resize(imagen_bgr, None, fx=escala, fy=escala,
                                interpolation=cv2.INTER_AREA)
    kernels = _kernels_para_escala(escala)

    H, W = imagen_bgr.shape[:2]

    # Máscara por color: unión de los rangos definidos para la referencia.
    rangos = rangos_hsv if rangos_hsv else _RANGOS_HSV_FALLBACK_LANNATE
    hsv    = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    union  = np.zeros((H, W), dtype=np.uint8)
    for lo, hi in rangos:
        union |= cv2.inRange(hsv, lo, hi)

    # OPEN (45 px físicos): aniquila reflejos, halos y puentes finos.
    limpia = cv2.morphologyEx(union, cv2.MORPH_OPEN, kernels["apertura"],
                              iterations=1)

    # Limpiar franjas LED del domo — caen en "blanco" (V alto, S bajo) y
    # sobreviven al OPEN porque son bandas anchas.  Detección adaptativa:
    # si una franja tiene >20 % de brillo concentrado, se pone a 0.
    limpia = _limpiar_franjas_led(
        limpia, franja_px=max(8, int(FRANJA_LED_PX * escala))
    )

    # Pre-cierre: une banner, cuerpo y franja roja cuando sus divisorias
    # de bajo contraste los dejan como blobs separados (alcance ~50 px
    # físicos: kernel base 25 × 2 iter, escalado).
    unido = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernels["pre_cierre"],
                             iterations=2)

    # Componentes conexas → blob mayor = label.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unido, connectivity=8)
    if n_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_mayor = 1 + int(np.argmax(areas))
    area_mayor = int(areas[idx_mayor - 1])

    if area_mayor < _AREA_MIN_ETIQUETA_FRAC * H * W:
        return None

    mask_principal = (labels == idx_mayor).astype(np.uint8) * 255

    # Cierre grande SOLO sobre el blob principal.  Rellena huecos
    # topográficos internos del banner (~200 px físicos).  No puede
    # introducir contaminación externa porque opera sobre una máscara
    # ya aislada.
    llena = cv2.morphologyEx(mask_principal, cv2.MORPH_CLOSE,
                             kernels["cierre"], iterations=3)

    mask = cv2.morphologyEx(llena, cv2.MORPH_CLOSE, kernels["pulido"])
    return mask


# ── Rectificación (cancela posición + rotación de la etiqueta) ───────────────

def _interseccion_lineas(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """
    Intersección de dos líneas parametrizadas como devuelve `cv2.fitLine`:
    cada una es (vx, vy, x0, y0) donde (vx, vy) es el vector dirección
    unitario y (x0, y0) un punto de la línea.

    Resuelve el sistema:  P0_1 + t1 · v1 = P0_2 + t2 · v2
    como matriz 2×2 con `np.linalg.solve` (lanza `LinAlgError` si las
    líneas son paralelas — los callers manejan eso).
    """
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    t = np.linalg.solve(A, b)
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1], dtype=np.float32)


def _ordenar_esquinas(pts: np.ndarray) -> np.ndarray:
    """
    Dado un array (4,2) de esquinas del bbox rotado, las ordena como
    tl, tr, br, bl (top-left, top-right, bottom-right, bottom-left).

    Reglas:
        - tl: suma x+y mínima (esquina más cercana al origen).
        - br: suma x+y máxima.
        - tr: x grande + y pequeño → diferencia (y-x) mínima.
        - bl: x pequeño + y grande → diferencia (y-x) máxima.

    Ojo: `np.diff(pts, axis=1)` devuelve y-x (segunda columna menos primera),
    no x-y.  Antes había un bug aquí que intercambiaba tr y bl, resultando
    en un warpPerspective reflejado sobre la diagonal tl-br.
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()   # y - x
    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right  (y-x minimo → y chico, x grande)
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left (y-x maximo → y grande, x chico)
    ], dtype=np.float32)


# Ancho (px) de la franja en cada borde del frame que consideramos "zona
# prohibida" — si la etiqueta realmente entró ahí, su máscara tendrá muchos
# píxeles a 255 en esa franja.
MARGEN_BORDE_FRAME = 20

# Umbral de corte (fracción del lado correspondiente): cuántos píxeles de la
# máscara deben tocar la franja de borde antes de declarar "etiqueta cortada".
# Con 5%, sombras/reflejos de ~2% (fuera_domo.jpg: 62 px en un lado de 3080)
# no disparan el rechazo, pero una etiqueta realmente partida (cientos/miles
# de píxeles apoyados sobre el borde) sí lo hace.
FRAC_MIN_BORDE_CORTE = 0.05

# NOTA sobre aspect ratio:  evaluamos inicialmente rechazar bboxes con aspect
# muy distinto al del canvas canónico, pero las etiquetas reales tienen
# aspect ~1.38 mientras el canvas es 1.75 — una validación numérica sobre
# el canvas rechaza casos legítimos.  En cambio, la validación de borde
# (abajo) captura el caso "etiqueta cortada" de forma directa y sin
# falsos positivos, así que ese es el único guardia pre-warp.


def _mascara_toca_frame(mask: np.ndarray,
                        margen: int = MARGEN_BORDE_FRAME,
                        frac_min: float = FRAC_MIN_BORDE_CORTE) -> bool:
    """
    True si la máscara tiene una cantidad significativa de píxeles dentro
    de la franja `margen` de algún borde del frame.

    Usamos el conteo de píxeles de la máscara, no los vértices del polígono
    aproximado del contorno: `cv2.CHAIN_APPROX_SIMPLE` puede dejar vértices
    espurios pegados al borde aunque la máscara real no llegue hasta ahí,
    generando falsos positivos en etiquetas perfectamente centradas.

    El umbral es por lado (no sobre el perímetro total) para que una etiqueta
    apoyada contra un único borde — el caso típico de "entró medio cortada"
    — se detecte aunque los otros tres bordes estén libres.
    """
    H, W = mask.shape[:2]

    # Cada franja tiene longitud = lado del frame (en el eje largo).  El umbral
    # se expresa como fracción de ese lado para que sea invariante a resolución.
    umbral_horizontal = int(frac_min * W)   # franjas superior/inferior
    umbral_vertical   = int(frac_min * H)   # franjas izquierda/derecha

    # Contamos píxeles de la máscara (>0) en cada franja. `np.count_nonzero`
    # sobre un slice es O(N) y extremadamente barato a estas resoluciones.
    tope_sup = np.count_nonzero(mask[:margen, :])
    tope_inf = np.count_nonzero(mask[H - margen:, :])
    tope_izq = np.count_nonzero(mask[:, :margen])
    tope_der = np.count_nonzero(mask[:, W - margen:])

    return bool(
        tope_sup > umbral_horizontal
        or tope_inf > umbral_horizontal
        or tope_izq > umbral_vertical
        or tope_der > umbral_vertical
    )


def rectificar_etiqueta(
    imagen_bgr: np.ndarray,
    target_w: int = CANONICAL_W,
    target_h: int = CANONICAL_H,
    rangos_hsv: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Recorta y endereza la etiqueta a un canvas de tamaño fijo.

    Pasos:
        1. segmentar_etiqueta() → máscara con `RANGOS_HSV_NO_DOME`.
        2. contorno → 4 esquinas (extremos diagonales) → bbox rotado.
        3. Validación pre-warp: el contorno no toca el borde del frame.
        4. Normalizar orientación apaisada.
        5. warpPerspective a (target_w, target_h).

    Retorna:
        (imagen_rectificada, None) si todo OK — np.ndarray (target_h, target_w, 3)
        (None, razon) si se rechazó.  `razon` ∈ {sin_segmentacion,
        contorno_vacio, contorno_diminuto, etiqueta_cortada}.

    Sobre el parámetro `rangos_hsv`:
        Se mantiene por compatibilidad pero se IGNORA.  Internamente
        usamos siempre `RANGOS_HSV_NO_DOME` (V ≥ 50) para detectar el
        PERÍMETRO del cartón completo.  Los rangos calibrados por
        referencia (banner verde, cuerpo blanco, franja roja) están
        afinados para verificar el CONTENIDO de la etiqueta, no el
        cartón.  En Lannate, los bordes del banner verde tienen verde
        más oscuro/amarillento que cae fuera del rango calibrado → la
        máscara queda ~400 px más estrecha en la zona del banner que
        en el cuerpo → las esquinas detectadas en la zona del banner
        están adentro del cartón físico → el warp produce trapecio
        aparente porque estira el top más que el bottom.  La
        verificación de paleta queda delegada al detector de celdas
        (DetectorROI) tras la rectificación, no acopla la segmentación.

    Por qué validamos que el contorno no toque el borde antes del warp:
        Si la etiqueta entra parcialmente fuera del FOV, el bbox captura
        sólo el trozo visible.  El warpPerspective estira ese trozo al
        canvas canónico → la red ve un "Lannate" completo y bien formado
        cuando en realidad falta la mitad superior.  La rectificación
        borra la evidencia del corte.  La validación atrapa ese caso.

    Corrección de distorsión del lente:
        Se asume que el frame de entrada YA viene undistortado.  La
        corrección se aplica en `Camara.capturar()` sobre el frame
        completo (4000×3000), antes del recorte.
    """
    # rangos_hsv se ignora — siempre usamos no-dome.  Ver docstring.
    rangos_hsv = RANGOS_HSV_NO_DOME

    # VARIANTE ROI: la segmentación corre a ESCALA_SEG (¼) — la máscara, el
    # contorno y las esquinas quedan en el espacio reducido y se re-escalan
    # al frame original justo antes del warp.  Ver comentario de ESCALA_SEG.
    mask = segmentar_etiqueta(imagen_bgr, rangos_hsv=rangos_hsv,
                              escala=ESCALA_SEG)
    if mask is None:
        return None, "sin_segmentacion"

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, "contorno_vacio"

    cnt = max(contornos, key=cv2.contourArea)
    # Umbral de área escalado (1000 px² a escala 1.0 → ×escala²).
    if cv2.contourArea(cnt) < 1000 * ESCALA_SEG ** 2:
        return None, "contorno_diminuto"

    # Validación: ¿la máscara apoya una fracción significativa contra el borde?
    #   → etiqueta parcialmente fuera del área de inspección.  El margen en px
    #   se escala; las fracciones de corte son relativas y no cambian.
    if _mascara_toca_frame(mask,
                           margen=max(3, int(MARGEN_BORDE_FRAME * ESCALA_SEG))):
        return None, "etiqueta_cortada"

    # Detección de las 4 esquinas del label por extremos diagonales del
    # contorno: tl=argmin(x+y), tr=argmin(y-x), br=argmax(x+y), bl=argmax(y-x).
    #
    # Por qué los extremos diagonales y no la intersección de líneas
    # extrapoladas:
    #   El cartón del label tiene esquinas físicas REDONDEADAS (radio
    #   ~10-30 px en Lannate).  Hay dos opciones para "esquina":
    #     (a) Punto del CONTORNO más cerca del rincón teórico → cae sobre
    #         el material del label (sobre el arco del redondeo).
    #     (b) Intersección de las líneas extrapoladas top/left, etc. →
    #         cae en el rincón geométrico afilado que NO existe físicamente,
    #         está FUERA del cartón sobre el dome negro.
    #   Con (b), `getPerspectiveTransform` mapea ese rincón geométrico
    #   FUERA del cartón a la esquina del canvas → la esquina del canvas
    #   queda OSCURA porque toma un píxel de dome.  Probado, descartado.
    #   Con (a), la rectificada pierde pocos píxeles de la esquina
    #   redondeada (esquinas se "cortan") pero todo el canvas es label.
    #
    # Sobre el "trapecio percibido" en Lannate:  El usuario suele percibir
    # que el label es más ancho arriba que abajo.  Mediciones del contorno
    # confirman que NO es físico (top body ancho ≈ bottom body, diferencia
    # 1-2 %): es ilusión óptica del DISEÑO IMPRESO — banner verde y
    # franja roja llegan a los bordes del cartón pero el cuerpo blanco
    # tiene márgenes blancos alrededor del contenido impreso.  No se
    # corrige en código porque el cartón sí es rectangular.
    pts = cnt.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    diag = np.array([
        pts[np.argmin(s)],   # tl
        pts[np.argmin(d)],   # tr
        pts[np.argmax(s)],   # br
        pts[np.argmax(d)],   # bl
    ], dtype=np.float32)

    ancho_sup  = float(np.linalg.norm(diag[1] - diag[0]))
    ancho_inf  = float(np.linalg.norm(diag[2] - diag[3]))
    alto_izq   = float(np.linalg.norm(diag[3] - diag[0]))
    alto_der   = float(np.linalg.norm(diag[2] - diag[1]))
    asim_horiz = abs(ancho_sup - ancho_inf) / max(ancho_sup, ancho_inf)
    asim_vert  = abs(alto_izq - alto_der)   / max(alto_izq, alto_der)

    # Umbral 8 %: deja pasar keystone real moderado; cae a minAreaRect
    # cuando hay contaminación local extrema (asim > 9-10 % en un solo
    # eje, típico de residuos de LED arrastrando una esquina).
    if asim_horiz <= 0.08 and asim_vert <= 0.08:
        orden = diag
    else:
        logging.debug(
            "rectificar_etiqueta: fallback a minAreaRect "
            "(asim_horiz=%.1f%%, asim_vert=%.1f%%)",
            asim_horiz * 100, asim_vert * 100,
        )
        orden = _ordenar_esquinas(
            cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        )

    # Re-escalar las esquinas del espacio reducido de segmentación al frame
    # original — el warp toma los píxeles de la imagen a resolución completa.
    orden = (orden / ESCALA_SEG).astype(np.float32)

    # Orientación: tl→tr horizontal, tl→bl vertical.  Si el lado superior es
    # más largo que el izquierdo, la etiqueta está apaisada (caso normal).
    d_superior = float(np.linalg.norm(orden[1] - orden[0]))
    d_izquierda = float(np.linalg.norm(orden[3] - orden[0]))
    es_apaisada = d_superior >= d_izquierda

    if es_apaisada:
        dest = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ], dtype=np.float32)
    else:
        # Si la etiqueta está "de canto", rotamos el destino 90°:
        # tl del bbox original va a la esquina superior izquierda del canvas
        # pero el lado largo se proyecta horizontal.
        dest = np.array([
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
            [0, 0],
        ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(orden, dest)
    rectificada = cv2.warpPerspective(
        imagen_bgr, M, (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rectificada, None


def rectificar_si_necesario(
    imagen_bgr: np.ndarray,
    rangos_hsv: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Rectifica SOLO si la imagen aún no lo está.  Si su tamaño coincide con
    el canvas canónico (CANONICAL_W × CANONICAL_H), se asume pre-rectificada
    y se devuelve sin modificar.  Esto ahorra un warp redundante — y evita
    el doble pase de interpolación — cuando el banco de entrenamiento
    guarda las muestras ya rectificadas.

    Útil en el pipeline de entrenamiento (DetectorROI.entrenar) donde el
    caller puede pasar tanto frames crudos de la cámara como imágenes ya
    pre-procesadas.
    """
    if (imagen_bgr.shape[1] == CANONICAL_W
            and imagen_bgr.shape[0] == CANONICAL_H):
        return imagen_bgr, None
    return rectificar_etiqueta(imagen_bgr, rangos_hsv=rangos_hsv)
