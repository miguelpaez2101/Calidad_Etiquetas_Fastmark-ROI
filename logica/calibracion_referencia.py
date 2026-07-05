"""
calibracion_referencia - Calibración por-referencia de rangos HSV
==================================================================

Cada referencia (Lannate, Harvanta, …) tiene una paleta distinta.
Este módulo concentra:

1. **Extracción de rangos HSV desde un ROI** sobre la maestra rectificada.
   El operador pinta un rectángulo sobre la región de un color (banner verde,
   cuerpo blanco, franja roja, etc.); de ese crop se calculan percentiles
   robustos por canal y se construye un rango (lo, hi) — o dos, en caso de
   wraparound del matiz (típico del rojo).

2. **Persistencia** de un archivo por referencia:
       referencias/<ref>/calibracion_color.json
   Se mantiene separado de meta.json porque puede crecer (stats por ROI)
   y conviene leerlo como bloque independiente.

3. **Helpers de estado**: `referencia_calibrada(ref_dir)` retorna True sólo si
   ese archivo existe — los callers usan esto para bloquear el botón
   "Entrenar" cuando una referencia recién creada aún no fue calibrada.
   (Esta variante no tiene OCR: no existe ocr_template.json.)

NOTA — wraparound del canal H:
    En OpenCV, H ∈ [0, 180] y rojo = 0 ≈ 180.  Si el ROI captura un color
    rojo puro, los píxeles tendrán H concentrado en dos extremos (≈[0,15]
    y ≈[165,180]).  La heurística aquí: si el span percentil-90 supera la
    mitad del rango (90 unidades), partimos en dos rangos circulares.  Para
    colores no-rojos (verde H≈65, azul H≈110, etc.) el span queda mucho
    menor y el rango es uno solo.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


# ── Parámetros de extracción HSV ────────────────────────────────────────────

# Percentiles robustos para definir min/max del rango aprendido.  Descartamos
# 5% por cada extremo: protege contra píxeles outlier (brillos especulares,
# bordes de letras) que sino estirarían el rango hasta capturar partes del
# dome o de regiones vecinas.
_PERC_LO = 5.0
_PERC_HI = 95.0

# Margen extra tras los percentiles.  Compensa fluctuaciones de iluminación
# entre la sesión de calibración y la de inspección.  Valores apretados —
# preferimos rechazar una buena por márgenes chicos a aceptar dome como
# "etiqueta" por márgenes amplios (caso real con cuerpo_blanco V_lo=22 que
# capturaba todo el fondo oscuro).
_MARGEN_H = 5     # de 180 (~2.8°)
_MARGEN_S = 15
_MARGEN_V = 15

# Pisos/topes absolutos sobre los rangos calculados.  Prioridad sobre los
# percentiles cuando éstos no son razonables.  Caso típico: el ROI
# "cuerpo_blanco" pintado sobre todo el cuerpo del label incluye TEXTO
# NEGRO del propio label (LANNATE, Methomyl, etc., V<50, ~2% del ROI).  Sin
# este piso, los percentiles bajos del ROI capturan ese texto y al aplicar
# el rango a la cámara se traga también el dome (V<40 también).
_V_LO_MIN = 60
_S_HI_MAX_ACROMATICO = 80  # ROI "blanco/gris": no permitir capturar colores

# Si la saturación media del ROI cae por debajo de este umbral, el ROI es
# acromático — el matiz H no aporta información (ruido aleatorio en píxeles
# desaturados).  Para esos ROIs (típicamente "blanco cuerpo" o "gris fondo"),
# el rango H se abre completo y la discriminación queda en S y V.
_S_MIN_PARA_H_VALIDO = 40

# Wraparound: para que el split en dos sub-rangos se dispare, debe haber
# AL MENOS este porcentaje de píxeles en cada extremo del círculo de matiz.
# Evita falsos splits cuando un único H está cerca del borde y unos pocos
# píxeles de ruido caen al otro lado.  5% es suficiente para detectar el
# rojo de Lannate (47-58% en H<30 y ~10% en H>150) sin falsos splits con
# colores no-wraparound.
_FRAC_MIN_WRAPAROUND = 0.05

# Filtros absolutos de píxeles "del color objetivo" antes de calcular
# percentiles.  El percentil_25 dinámico falla cuando el ROI tiene mucho
# contraste — si la franja roja tiene 30% de iconos blancos saturados al
# borde, S_perc25 cae a ≈8 y casi nada se descarta, abriendo el rango a
# todos los colores presentes.  Los pisos absolutos son específicos al
# tipo de ROI y NO dependen de la distribución del crop.
_S_MIN_CROMATICO = 60     # píxeles "del color": S >= 60 (texto/blanco fuera)
_V_MIN_CROMATICO = 50     # píxeles "del color": V >= 50 (texto negro fuera)
_V_MAX_CROMATICO = 240    # píxeles "del color": V <= 240 (brillos fuera)
_S_MAX_ACROMATICO = 50    # ROI "blanco/gris": S <= 50 (color real fuera)
_V_MIN_ACROMATICO = 100   # ROI "blanco/gris": V >= 100 (texto negro fuera)
_V_MAX_ACROMATICO = 240   # ROI "blanco/gris": V <= 240 (brillos especulares fuera)
# Si el filtro absoluto deja muy pocos píxeles (<10% del ROI), relajamos al
# fallback dinámico con percentil_25 para no quedarnos sin datos.
_FRAC_MIN_PIXELS_FILTRADOS = 0.10
_PERC_FILTRO_DINAMICO = 25.0


def extraer_rango_hsv(
    imagen_rect_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict:
    """
    Calcula los rangos HSV característicos de un ROI sobre la maestra rectificada.

    Parámetros:
        imagen_rect_bgr: canvas canónico BGR (cv_util.CANONICAL_W×CANONICAL_H).
        bbox: (x, y, w, h) en coordenadas del canvas.

    Retorna:
        dict con:
            "rangos": list de pares ([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]).
                Una entrada en colores normales; dos en colores con wraparound.
            "stats": {"h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
                      "n_pixels_filtrado", "n_pixels_total"}
                Media y desviación de cada canal sobre los píxeles que se
                usaron para calcular los rangos (post-filtro), útil para
                auditoría y depuración.

    Pipeline interno:
      1. Decide si el ROI es cromático (S_media ≥ 40) o acromático.
      2. Pre-filtra píxeles: en cromáticos, descarta el cuartil más gris (S
         baja); en acromáticos, descarta el cuartil más oscuro (V baja).
         Razón: ROIs grandes del operador suelen incluir píxeles que no son
         del color objetivo (bordes, marcas, espacios) y meterlos en el
         cómputo abre el rango a nivel del fondo del domo.
      3. Calcula percentiles 5-95 + margen sobre los píxeles representativos.
      4. Aplica pisos: V_lo ≥ 60 (bloquea fondo del domo); en acromáticos
         S_hi ≤ 80 (no permite capturar colores legítimos del label).
      5. Si el rango H tiene wraparound real (≥10 % de píxeles en cada
         extremo del círculo), parte en dos sub-rangos circulares.
    """
    x, y, w, h = bbox
    crop = imagen_rect_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"extraer_rango_hsv: ROI vacío {bbox}")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H_all = hsv[:, :, 0].ravel().astype(np.int32)
    S_all = hsv[:, :, 1].ravel().astype(np.int32)
    V_all = hsv[:, :, 2].ravel().astype(np.int32)
    n_total = len(H_all)

    # Decisión cromático vs acromático.  Usamos la mediana (no la media)
    # para que un ROI con muchos píxeles oscuros (incluyendo bordes negros)
    # no nos induzca a tratarlo como acromático cuando el color dominante sí
    # tiene saturación.
    s_mediana = float(np.median(S_all))
    es_cromatico = s_mediana >= _S_MIN_PARA_H_VALIDO

    # Pre-filtro: quedarnos sólo con los píxeles "del color objetivo".
    # Pisos absolutos específicos al tipo de ROI:
    #  - Cromático (banner verde, franja roja): S alto + V medio descarta
    #    texto blanco (S~10), texto negro (V<50), brillos especulares (V>240).
    #  - Acromático (cuerpo blanco): S bajo + V medio descarta colores
    #    legítimos del label (logos, iconos), texto negro y brillos.
    if es_cromatico:
        keep = ((S_all >= _S_MIN_CROMATICO)
                & (V_all >= _V_MIN_CROMATICO)
                & (V_all <= _V_MAX_CROMATICO))
    else:
        keep = ((S_all <= _S_MAX_ACROMATICO)
                & (V_all >= _V_MIN_ACROMATICO)
                & (V_all <= _V_MAX_ACROMATICO))

    # Si los pisos absolutos dejan muy pocos píxeles, el ROI es atípico
    # (color débil, iluminación particular).  Caemos al fallback dinámico
    # de percentil_25 sobre el canal dominante para no quedarnos sin datos.
    if keep.sum() < _FRAC_MIN_PIXELS_FILTRADOS * n_total:
        if es_cromatico:
            s_corte = np.percentile(S_all, _PERC_FILTRO_DINAMICO)
            keep = S_all >= s_corte
        else:
            v_corte = np.percentile(V_all, _PERC_FILTRO_DINAMICO)
            keep = V_all >= v_corte

    H = H_all[keep]
    S = S_all[keep]
    V = V_all[keep]

    s_mean = float(S.mean())
    v_mean = float(V.mean())
    h_mean = float(H.mean())

    s_lo = int(np.clip(np.percentile(S, _PERC_LO) - _MARGEN_S, 0, 255))
    s_hi = int(np.clip(np.percentile(S, _PERC_HI) + _MARGEN_S, 0, 255))
    v_lo = int(np.clip(np.percentile(V, _PERC_LO) - _MARGEN_V, 0, 255))
    v_hi = int(np.clip(np.percentile(V, _PERC_HI) + _MARGEN_V, 0, 255))

    # Pisos absolutos: el dome del prototipo cae en V<40, así que cualquier
    # rango que baje V_lo de 60 garantiza falsos positivos por capturar
    # sombras del fondo.  Aplicamos siempre, no es opcional.
    v_lo = max(v_lo, _V_LO_MIN)

    # ── H: tres caminos según saturación y dispersión ────────────────────────
    if not es_cromatico:
        # Acromático: H carece de significado.  Abrimos el rango entero y
        # apretamos S como discriminante (s_hi acotado para que el rango
        # "blanco" no devore colores reales con S~80-100).
        s_hi = min(s_hi, _S_HI_MAX_ACROMATICO)
        rangos = [
            ([0, s_lo, v_lo], [180, s_hi, v_hi]),
        ]
    else:
        # ¿Wraparound real?  Solo si AMBOS extremos contienen ≥10% de los
        # píxeles del color (post-filtro).  Antes el criterio era el span
        # del percentil, que se disparaba con falsos rojos cuando había
        # cualquier píxel cerca de H=0 y otro cerca de H=180.
        n_low  = int((H <  30).sum())
        n_high = int((H > 150).sum())
        n_keep = len(H)
        es_wraparound = (
            n_keep > 0
            and n_low  >= _FRAC_MIN_WRAPAROUND * n_keep
            and n_high >= _FRAC_MIN_WRAPAROUND * n_keep
        )

        if es_wraparound:
            # Tomamos un sub-rango compacto en cada extremo del círculo.
            # Usamos cuartiles (75% / 25%) para que cada lado quede angosto
            # alrededor de su pico — evita que el wraparound se transforme
            # en "casi todo H".
            h_low_max  = int(np.clip(np.percentile(H[H <  90], 75) + _MARGEN_H, 0, 180))
            h_high_min = int(np.clip(np.percentile(H[H >= 90], 25) - _MARGEN_H, 0, 180))
            rangos = [
                ([0,          s_lo, v_lo], [h_low_max,  s_hi, v_hi]),
                ([h_high_min, s_lo, v_lo], [180,        s_hi, v_hi]),
            ]
        else:
            h_lo = int(np.clip(np.percentile(H, _PERC_LO) - _MARGEN_H, 0, 180))
            h_hi = int(np.clip(np.percentile(H, _PERC_HI) + _MARGEN_H, 0, 180))
            rangos = [
                ([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]),
            ]

    return {
        "rangos": [list(r) for r in rangos],
        "stats": {
            "h_mean": h_mean,
            "h_std":  float(H.std()),
            "s_mean": s_mean,
            "s_std":  float(S.std()),
            "v_mean": v_mean,
            "v_std":  float(V.std()),
            "n_pixels_filtrado": int(len(H)),
            "n_pixels_total":    int(n_total),
            "es_cromatico":      bool(es_cromatico),
        },
    }


def rangos_planos(calibracion_color: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Aplana la calibración color a la estructura que espera `segmentar_etiqueta`:
    una lista de (lo, hi) en arrays uint8 listos para `cv2.inRange`.

    La unión de las máscaras `inRange` por cada par equivale a la máscara
    conjunta de todas las regiones del label — banner verde ∪ cuerpo blanco
    ∪ franja roja, etc.
    """
    salida: list[tuple[np.ndarray, np.ndarray]] = []
    for roi in calibracion_color.get("rois", []):
        for lo, hi in roi.get("rangos", []):
            salida.append(
                (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            )
    return salida


# ── Persistencia ────────────────────────────────────────────────────────────

_NOMBRE_COLOR = "calibracion_color.json"
_NOMBRE_MAESTRA = "plantilla_maestra.jpg"


def ruta_calibracion_color(carpeta_referencia: str | Path) -> Path:
    return Path(carpeta_referencia) / _NOMBRE_COLOR


def ruta_maestra(carpeta_referencia: str | Path) -> Path:
    return Path(carpeta_referencia) / _NOMBRE_MAESTRA


def cargar_calibracion_color(carpeta_referencia: str | Path) -> dict | None:
    ruta = ruta_calibracion_color(carpeta_referencia)
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("calibracion_color.json inválido en %s: %s",
                        carpeta_referencia, e)
        return None


def guardar_calibracion_color(
    carpeta_referencia: str | Path,
    rois: list[dict],
) -> bool:
    """
    Guarda la calibración de color de una referencia.

    Parámetros:
        carpeta_referencia: carpeta de la referencia.
        rois: lista de dicts {"nombre": str, "bbox": [x,y,w,h],
                              "rangos": [...], "stats": {...}}
              tal como los devuelve `extraer_rango_hsv` enriquecidos con
              nombre y bbox.
    """
    payload = {
        "version": "1.0",
        "creada":  datetime.now().isoformat(timespec="seconds"),
        "rois":    rois,
    }
    ruta = ruta_calibracion_color(carpeta_referencia)
    try:
        ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return True
    except OSError as e:
        logging.error("No se pudo escribir %s: %s", ruta, e)
        return False


def referencia_calibrada(carpeta_referencia: str | Path) -> bool:
    """
    True si existe calibracion_color.json — único archivo de calibración
    en esta variante (sin OCR).

    Es la pre-condición para habilitar el entrenamiento del modelo de
    color: sin la paleta calibrada no hay rangos de referencia para la
    verificación de contenido.
    """
    return ruta_calibracion_color(carpeta_referencia).exists()


# ── Calibración automática (k-means HSV sobre la maestra rectificada) ───────
#
# El operador ya no pinta ROIs.  El sistema detecta automáticamente las
# regiones de color del label aplicando k-means sobre el HSV de la maestra
# rectificada.  Cada cluster genera una "ROI virtual":
#   - bbox: el rectángulo que envuelve los píxeles de ese cluster (descartando
#     ruido aislado con un erode pequeño).
#   - rangos: percentiles 5/95 del HSV de ese cluster + márgenes + pisos
#     absolutos (mismo cálculo que `extraer_rango_hsv` pero a partir de los
#     píxeles del cluster en vez de un rectángulo dibujado).
#   - stats: medias/std para auditoría y verificación de contenido.
#   - nombre: lookup por H_mean (verde, rojo, azul, etc.) o "color_N".

# K-means: rango de k a probar.  El criterio es elegir el MAYOR k tal que
# todos los clusters tengan al menos `_FRAC_MIN_CLUSTER` del label.  Más
# clusters = paleta más granular; pero clusters chicos generan rangos
# inestables.  3-5 cubre el caso típico (etiqueta con 3-4 colores grandes
# + 1 accent ocasional).
_K_RANGE = (3, 4, 5)
_FRAC_MIN_CLUSTER = 0.05
# Submuestreo para entrenar k-means: k-means converge bien con ~50-100 k
# muestras aunque el canvas tenga más píxeles.  Acota el tiempo de fit
# sin afectar la calidad de los centros.
_N_MUESTRA_KMEANS = 80_000

# Lookup de nombres de color por rango de H (en OpenCV, H ∈ [0, 180]).
# Cuando el cluster es acromático (S baja) usamos "blanco" o "gris" según
# V_mean.  Para colores cromáticos elegimos por la franja de H_mean.
def _nombre_color(h_mean: float, s_mean: float, v_mean: float) -> str:
    if s_mean < 40:
        return "blanco" if v_mean > 150 else "gris"
    # Cromático: nombre por matiz.  Rangos amplios para que cubran bien
    # cualquier H_mean.
    if h_mean < 10 or h_mean >= 165:
        return "rojo"
    if h_mean < 25:
        return "naranja"
    if h_mean < 35:
        return "amarillo"
    if h_mean < 85:
        return "verde"
    if h_mean < 125:
        return "azul"
    if h_mean < 150:
        return "violeta"
    return "magenta"


def _rango_desde_pixels(H, S, V, es_cromatico: bool) -> dict:
    """
    Calcula rangos HSV (con wraparound si aplica) a partir de un conjunto de
    píxeles ya filtrados.  Misma lógica que `extraer_rango_hsv` pero opera
    sobre arrays de píxeles, no sobre un bbox de imagen.
    """
    s_lo = int(np.clip(np.percentile(S, _PERC_LO) - _MARGEN_S, 0, 255))
    s_hi = int(np.clip(np.percentile(S, _PERC_HI) + _MARGEN_S, 0, 255))
    v_lo = int(np.clip(np.percentile(V, _PERC_LO) - _MARGEN_V, 0, 255))
    v_hi = int(np.clip(np.percentile(V, _PERC_HI) + _MARGEN_V, 0, 255))
    v_lo = max(v_lo, _V_LO_MIN)

    if not es_cromatico:
        s_hi = min(s_hi, _S_HI_MAX_ACROMATICO)
        return {
            "rangos": [[[0, s_lo, v_lo], [180, s_hi, v_hi]]],
        }

    n = len(H)
    n_low  = int((H <  30).sum())
    n_high = int((H > 150).sum())
    es_wrap = (n_low  >= _FRAC_MIN_WRAPAROUND * n
               and n_high >= _FRAC_MIN_WRAPAROUND * n)
    if es_wrap:
        h_low_max  = int(np.clip(np.percentile(H[H <  90], 75) + _MARGEN_H, 0, 180))
        h_high_min = int(np.clip(np.percentile(H[H >= 90], 25) - _MARGEN_H, 0, 180))
        return {
            "rangos": [
                [[0,          s_lo, v_lo], [h_low_max,  s_hi, v_hi]],
                [[h_high_min, s_lo, v_lo], [180,        s_hi, v_hi]],
            ],
        }
    h_lo = int(np.clip(np.percentile(H, _PERC_LO) - _MARGEN_H, 0, 180))
    h_hi = int(np.clip(np.percentile(H, _PERC_HI) + _MARGEN_H, 0, 180))
    return {
        "rangos": [[[h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]]],
    }


def _bbox_de_cluster(mask_cluster: np.ndarray) -> list[int]:
    """
    Bbox del componente conexo MAYOR del cluster, tras un erode de 5 px que
    descarta píxeles aislados (texto disperso, ruido).  Devuelve [x, y, w, h].

    Si la erosion vacía la máscara (cluster fragmentado), cae al bbox global
    de la máscara sin filtrar — en ese caso bbox abarca todo.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.morphologyEx(mask_cluster, cv2.MORPH_ERODE, kernel)
    if eroded.sum() == 0:
        eroded = mask_cluster
    n, _, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n <= 1:
        ys, xs = np.where(mask_cluster > 0)
        if len(xs) == 0:
            return [0, 0, mask_cluster.shape[1], mask_cluster.shape[0]]
        return [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[idx]
    return [int(x), int(y), int(w), int(h)]


def calibrar_automaticamente(maestra_rectificada_bgr: np.ndarray) -> list[dict]:
    """
    Detecta automáticamente la paleta de la etiqueta sobre la maestra
    rectificada y devuelve la lista de ROIs virtuales lista para guardar
    como `calibracion_color.json`.

    Pipeline:
        1. Pre-filtro de píxeles: descarta texto/sombras (V<50) y brillos
           especulares (V>240).  Sin esto k-means agrupa el texto negro
           con los blancos (ambos S baja).
        2. Mapeo H circular: H → (cos, sin) en el espacio de features,
           para que el rojo wraparound (H≈0 ≈ H≈180) quede en un solo
           cluster.  S y V se escalan al mismo rango aproximado.
        3. K-means con k=3,4,5; elige el mayor k tal que todos los
           clusters tengan ≥ 5 % del label filtrado.
        4. Fusión de clusters redundantes: si dos clusters acabaron con
           el mismo `_nombre_color`, se unen en uno.
        5. Para cada cluster:
             - filtra píxeles representativos (pisos absolutos),
             - calcula rangos HSV con percentiles + margen + pisos,
             - calcula bbox del componente conexo mayor.
        6. Devuelve `[{"nombre", "bbox", "rangos", "stats"}, …]`.
    """
    h_canvas, w_canvas = maestra_rectificada_bgr.shape[:2]
    hsv = cv2.cvtColor(maestra_rectificada_bgr, cv2.COLOR_BGR2HSV)
    pixels_full = hsv.reshape(-1, 3).astype(np.float32)
    n_full = len(pixels_full)

    # Pre-filtro: descartamos texto negro y sombras (V<50) y brillos
    # especulares (V>240).  Esos no son parte de la "paleta" del label —
    # son artefactos de impresión/iluminación que solo confunden a k-means.
    V_full = pixels_full[:, 2]
    mask_paleta = (V_full >= 50) & (V_full <= 240)
    idx_paleta = np.where(mask_paleta)[0]
    pixels = pixels_full[idx_paleta]
    n_total = len(pixels)
    if n_total < 0.1 * n_full:
        # Iluminación atípica — no podemos calibrar con confianza.
        raise RuntimeError(
            f"Maestra con muy pocos píxeles válidos ({n_total/n_full*100:.1f}%) — "
            "verifica la iluminación del domo."
        )

    # Espacio de features para k-means: (cos(H), sin(H), S, V).
    # El factor 90 escala el círculo de H al rango ±90 (similar a S/V que
    # son [0, 255]) — sin esto, las diferencias en H pesan demasiado poco
    # frente a S/V y k-means agrupa por brillo en vez de por color.
    H_rad = pixels[:, 0] * (np.pi / 90.0)  # H ∈ [0,180] → ángulo [0, 2π]
    cos_h = np.cos(H_rad) * 90.0
    sin_h = np.sin(H_rad) * 90.0
    features = np.column_stack([cos_h, sin_h, pixels[:, 1], pixels[:, 2]])

    # Submuestrear para acelerar el fit.
    rng = np.random.default_rng(42)
    if n_total > _N_MUESTRA_KMEANS:
        idx_sample = rng.choice(n_total, _N_MUESTRA_KMEANS, replace=False)
        muestra = features[idx_sample]
    else:
        muestra = features

    # Probar varios k.
    mejor_k = None
    mejor_etiquetas = None
    for k in _K_RANGE:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(muestra)
        etiquetas = km.predict(features)
        fracciones = np.bincount(etiquetas, minlength=k) / n_total
        if (fracciones >= _FRAC_MIN_CLUSTER).all():
            mejor_k = k
            mejor_etiquetas = etiquetas
        else:
            break
    if mejor_k is None:
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        km.fit(muestra)
        mejor_etiquetas = km.predict(features)
        mejor_k = 3

    # Mapeo de etiquetas al canvas full (los pixels filtrados son un subset).
    etiquetas_full = -np.ones(n_full, dtype=np.int32)
    etiquetas_full[idx_paleta] = mejor_etiquetas

    logging.info(
        "calibrar_automaticamente: k=%d, fracciones=%s",
        mejor_k,
        np.bincount(mejor_etiquetas, minlength=mejor_k) / n_total,
    )

    # ── Pre-procesar cada cluster en datos crudos ─────────────────────────────
    # Generamos información intermedia de cada cluster (nombre + stats + bbox)
    # antes de fusionar, para poder identificar duplicados por nombre.
    fracs_paleta = np.bincount(mejor_etiquetas, minlength=mejor_k) / n_total
    fracs_full   = np.array([
        (etiquetas_full == k).sum() / n_full for k in range(mejor_k)
    ])
    orden = np.argsort(-fracs_paleta)

    pre_clusters: list[dict] = []
    for k in orden:
        idx_cluster_paleta = (mejor_etiquetas == k)
        cluster_pixels = pixels[idx_cluster_paleta]
        if len(cluster_pixels) == 0:
            continue
        H = cluster_pixels[:, 0].astype(np.int32)
        S = cluster_pixels[:, 1].astype(np.int32)
        V = cluster_pixels[:, 2].astype(np.int32)

        s_mediana = float(np.median(S))
        es_cromatico = s_mediana >= _S_MIN_PARA_H_VALIDO

        if es_cromatico:
            keep = ((S >= _S_MIN_CROMATICO)
                    & (V >= _V_MIN_CROMATICO)
                    & (V <= _V_MAX_CROMATICO))
        else:
            keep = ((S <= _S_MAX_ACROMATICO)
                    & (V >= _V_MIN_ACROMATICO)
                    & (V <= _V_MAX_ACROMATICO))
        if keep.sum() < _FRAC_MIN_PIXELS_FILTRADOS * len(H):
            keep = np.ones_like(H, dtype=bool)
        H_f, S_f, V_f = H[keep], S[keep], V[keep]
        if len(H_f) == 0:
            continue

        # Para H_mean en colores con wraparound del rojo, la media aritmética
        # da un valor central espurio (~90).  Usamos la media circular.
        if es_cromatico:
            ang = H_f * (np.pi / 90.0)
            h_mean = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                            * 90.0 / np.pi) % 180.0)
        else:
            h_mean = float(H_f.mean())
        s_mean = float(S_f.mean())
        v_mean = float(V_f.mean())

        pre_clusters.append({
            "k": int(k),
            "nombre_base": _nombre_color(h_mean, s_mean, v_mean),
            "es_cromatico": es_cromatico,
            "h_mean": h_mean, "s_mean": s_mean, "v_mean": v_mean,
            "H_f": H_f, "S_f": S_f, "V_f": V_f,
            "fraccion": float(fracs_full[k]),
        })

    # ── Fusionar clusters con el mismo nombre ────────────────────────────────
    # Es común que k-means con k=4-5 divida un mismo color en dos sub-clusters
    # (p.ej. "blanco brillante" y "blanco sombreado").  Si nuestro lookup
    # `_nombre_color` les asigna el mismo nombre, los unificamos: concatenamos
    # los píxeles y recomputamos rangos + bbox sobre el conjunto.
    fusionados: dict[str, dict] = {}
    ks_de: dict[str, list[int]] = {}
    for pc in pre_clusters:
        nombre = pc["nombre_base"]
        if nombre in fusionados:
            f = fusionados[nombre]
            f["H_f"] = np.concatenate([f["H_f"], pc["H_f"]])
            f["S_f"] = np.concatenate([f["S_f"], pc["S_f"]])
            f["V_f"] = np.concatenate([f["V_f"], pc["V_f"]])
            f["fraccion"] += pc["fraccion"]
            ks_de[nombre].append(pc["k"])
        else:
            fusionados[nombre] = dict(pc)
            ks_de[nombre] = [pc["k"]]

    # ── Construir ROIs finales ────────────────────────────────────────────────
    rois: list[dict] = []
    for nombre, fc in fusionados.items():
        H_f, S_f, V_f = fc["H_f"], fc["S_f"], fc["V_f"]
        rango_dict = _rango_desde_pixels(H_f, S_f, V_f, fc["es_cromatico"])

        # Bbox: unión de los pixels de todos los k-clusters que se fusionaron
        # bajo este nombre, luego erode + componente mayor.
        mask_cluster = np.zeros((h_canvas, w_canvas), dtype=np.uint8)
        for k_orig in ks_de[nombre]:
            mask_cluster |= ((etiquetas_full == k_orig)
                             .reshape(h_canvas, w_canvas)
                             .astype(np.uint8) * 255)
        bbox = _bbox_de_cluster(mask_cluster)

        # Recalcular medias sobre el conjunto fusionado (ya correctas para H
        # circular gracias al cálculo en pre_clusters).
        if fc["es_cromatico"]:
            ang = H_f * (np.pi / 90.0)
            h_mean = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                            * 90.0 / np.pi) % 180.0)
        else:
            h_mean = float(H_f.mean())

        rois.append({
            "nombre": nombre,
            "bbox":   bbox,
            "rangos": rango_dict["rangos"],
            "stats": {
                "h_mean": h_mean,
                "h_std":  float(H_f.std()),
                "s_mean": float(S_f.mean()),
                "s_std":  float(S_f.std()),
                "v_mean": float(V_f.mean()),
                "v_std":  float(V_f.std()),
                "n_pixels_filtrado": int(len(H_f)),
                "es_cromatico":      bool(fc["es_cromatico"]),
                "fraccion_label":    fc["fraccion"],
            },
        })

    # Orden final por fracción descendente (más grande primero, útil para UI).
    rois.sort(key=lambda r: -r["stats"]["fraccion_label"])
    return rois
