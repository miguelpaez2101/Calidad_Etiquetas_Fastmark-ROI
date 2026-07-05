"""
detector_roi - Detección de manchones y desvíos de color por celdas
====================================================================

Reemplazo ligero de PatchCore para la variante "solo ROI" del prototipo:
detecta manchas, oclusiones y cambios de color comparando estadísticas HSV
por CELDA contra las aprendidas de un banco de etiquetas buenas.  Todo en
CPU con numpy — sin torch, sin GPU, sin OOM.  Inferencia < 10 ms sobre el
canvas reducido (la latencia total la domina la rectificación).

Por qué celdas y no ROIs manuales (generaliza a firma_color.py):
    Una media HSV sobre una ROI grande diluye un manchón local (una mancha
    de 5 mm en el cuerpo blanco mueve la media del ROI < 1 unidad).  Con
    celdas de CELDA_PX (16 px ≈ 3.4 mm sobre el label) el manchón domina
    las celdas que toca y su z-score se dispara.  La grilla además
    LOCALIZA el defecto (qué celdas fallaron) sin costo extra.

Anclaje espacial — por qué esto atrapa el caso "cinta negra":
    PatchCore comparaba parches contra TODO el coreset: los parches
    oscuros de una cinta negra "matcheaban" con letras oscuras legítimas
    de otra zona y daba falso OK.  Aquí cada celda se compara SOLO contra
    su propia historia en esa posición: una celda que siempre fue blanca
    y llega negra dispara sin importar qué haya en el resto del label.

Features por celda (4 canales):
    cos(H)·90, sin(H)·90, S, V — H en coordenadas circulares para que el
    rojo (wraparound 0≈180) no infle la varianza (misma técnica que la
    calibración k-means).  El z-score normaliza por canal, así que la
    escala relativa de los canales no afecta; los pisos de σ sí se fijan
    por canal.

Calibración del umbral (leave-one-out):
    Con n imágenes en el banco, el score de cada imagen se calcula contra
    las estadísticas de las OTRAS n-1 (media y varianza LOO vectorizadas
    a partir de suma y suma de cuadrados).  Sin LOO, cada imagen se
    compararía contra estadísticas que la incluyen → scores optimistas →
    umbral demasiado bajo → falsos rechazos en producción.
    umbral = mediana(scores_loo) + IQR_MULTIPLICADOR · IQR (misma
    filosofía validada en el proyecto original).

Limitaciones conocidas (a propósito — ver CLAUDE.md de esta variante):
    - No detecta texto INCORRECTO con la misma paleta (lote/fecha mal
      impresos): eso requiere OCR (versión completa del proyecto).
    - No detecta defectos finos de textura (rayones delgados, blur).
    - Sensible a cambios de iluminación: el domo difuso y el banco
      capturan la tolerancia real; si se cambia la luz, reentrenar.
"""

import logging
import os
import pickle

import cv2
import numpy as np

from logica.cv_util import (rectificar_si_necesario, imread_safe,
                            CANONICAL_W, CANONICAL_H)

# ── Hiperparámetros ──────────────────────────────────────────────────────────

# Lado de la celda en px sobre el canvas canónico.  CANONICAL_W y CANONICAL_H
# deben ser múltiplos exactos (1024 = 64·16, 576 = 36·16 → grilla 64×36 =
# 2304 celdas).  Celda de 16 px ≈ 3.4 mm sobre el label real (216 mm de
# ancho): una mancha de 5 mm cubre al menos una celda completa.
CELDA_PX = 16

# Guarda dura: si el canvas no es múltiplo de la celda, _features_celdas
# recortaría en silencio la franja derecha/inferior sobrante → zona CIEGA
# del detector sin ningún aviso.  Mejor reventar al importar con un mensaje
# claro que inspeccionar con un punto ciego.
if CANONICAL_W % CELDA_PX or CANONICAL_H % CELDA_PX:
    raise ImportError(
        f"detector_roi: el canvas canónico {CANONICAL_W}x{CANONICAL_H} debe "
        f"ser múltiplo de CELDA_PX={CELDA_PX} en ambos ejes — ajustar "
        "cv_util.CANONICAL_* o CELDA_PX (y reentrenar todas las referencias)."
    )

# Score de imagen = media de las TOP_K celdas con mayor z-score.  Con K=4 un
# manchón real (varias celdas disparadas) domina el score, pero una única
# celda ruidosa (polvo en el sensor, borde de esquina redondeada) no
# rechaza sola la etiqueta.
TOP_K = 4

# umbral = mediana + IQR_MULTIPLICADOR · IQR sobre los scores LOO del banco.
IQR_MULTIPLICADOR = 3.0

# Pisos de σ por canal de feature (cosH·90, sinH·90, S, V).  Un banco muy
# uniforme colapsa σ→0 y cualquier fluctuación de ruido daría z enormes.
# Valores heredados de la filosofía de firma_color.py (SIGMA_MIN_*).
SIGMA_MIN = np.array([1.5, 1.5, 2.0, 2.5], dtype=np.float32)

# Mínimo de imágenes para calibrar umbral con LOO.  Por debajo se usa el
# fallback score_max · FACTOR_FALLBACK (igual que hacía el detector viejo
# con n=1).
N_MIN_LOO = 3
FACTOR_FALLBACK = 2.0

# ── Pre-check de presencia (¿hay una etiqueta de esta referencia?) ──────────
# Histograma HSV global reducido + brillo medio, aprendidos del banco.
# Atrapa "domo vacío pero brillante", "etiqueta de otra referencia" y
# fallos groseros de captura ANTES de mirar celdas (mensaje más claro
# para el operador que 500 celdas disparadas).
HIST_BINS_HSV        = (8, 4, 4)   # H×S×V = 128 bins
HIST_DIST_FACTOR_MAX = 1.5         # χ² > 1.5× la peor del banco → paleta ajena
BRILLO_SIGMAS        = 6.0
# Piso para la peor distancia χ² del banco: un banco muy homogéneo (capturas
# casi idénticas) colapsa hist_dist_max a ~1e-4 y el tope 1.5× rechazaría
# hasta capturas legítimas por fluctuación de compresión/ruido.  Una paleta
# realmente ajena da χ² en el orden de 0.3+ (caso "verde claro mala" del
# proyecto original: 0.369), así que un tope mínimo de 0.02·1.5=0.03 no
# debilita el pre-check.
PISO_DIST_HIST       = 0.02

# Versión del esquema de persistencia.  El .pkl guarda canvas y celda y
# `cargar` los valida — lección aprendida del proyecto original (schema v6):
# un modelo entrenado con otro canvas produciría scores inválidos en silencio.
PKL_VERSION = 1


class DetectorROI:
    """
    Detector de manchones/color por celdas.  API paralela a la del
    DetectorPatchCore original para que Inspector y MainWindow cambien lo
    mínimo: entrenar(rutas, callback) / guardar / cargar / inspeccionar /
    propiedades `entrenado` y `umbral` / set_calibracion.
    """

    def __init__(self):
        self._mu          = None   # (ny, nx, 4) media por celda/canal
        self._sigma       = None   # (ny, nx, 4) std por celda/canal (con pisos)
        self._umbral      = None   # float
        self._firma       = None   # dict presencia: hist/brillo
        self._rois        = []     # ROIs semánticas de la calibración (nombres)
        self._n_imagenes  = 0

    # ── Estado ───────────────────────────────────────────────────────────────

    @property
    def entrenado(self) -> bool:
        return (self._mu is not None and self._sigma is not None
                and self._umbral is not None)

    @property
    def umbral(self) -> float | None:
        return self._umbral

    @umbral.setter
    def umbral(self, valor: float):
        self._umbral = float(valor)

    def set_calibracion(self, rois: list | None):
        """
        Recibe las ROIs semánticas de `calibracion_color.json`
        ([{"nombre", "bbox", ...}, …], bbox en coords del canvas canónico).
        Solo se usan para NOMBRAR las celdas defectuosas en los mensajes
        ("mancha en banner verde") — la detección es por grilla y no
        depende de ellas.
        """
        self._rois = list(rois) if rois else []

    # ── Features ─────────────────────────────────────────────────────────────

    @staticmethod
    def _features_celdas(imagen_bgr: np.ndarray) -> np.ndarray:
        """
        (H, W, 3) BGR → (ny, nx, 4) float32 con la media por celda de
        (cos H·90, sin H·90, S, V).  Vectorizado con reshape — sin bucles.
        """
        hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32) * (np.pi / 90.0)  # H∈[0,180] → rad
        feats = np.stack([
            np.cos(h) * 90.0,
            np.sin(h) * 90.0,
            hsv[:, :, 1].astype(np.float32),
            hsv[:, :, 2].astype(np.float32),
        ], axis=-1)                                            # (H, W, 4)

        H, W = feats.shape[:2]
        ny, nx = H // CELDA_PX, W // CELDA_PX
        # (ny, CELDA, nx, CELDA, 4) → media sobre los ejes de píxel.
        celdas = feats[:ny * CELDA_PX, :nx * CELDA_PX].reshape(
            ny, CELDA_PX, nx, CELDA_PX, 4
        ).mean(axis=(1, 3))
        return celdas.astype(np.float32)

    @staticmethod
    def _firma_imagen(imagen_bgr: np.ndarray) -> dict:
        """Histograma HSV normalizado + brillo medio (para presencia)."""
        hsv  = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS_HSV),
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1)
        return {
            "hist":   hist.flatten().astype(np.float32),
            "brillo": float(hsv[:, :, 2].mean()),
        }

    @staticmethod
    def _dist_chi2(h1: np.ndarray, h2: np.ndarray) -> float:
        denom = h1 + h2
        denom[denom == 0] = 1e-9
        return float(0.5 * np.sum((h1 - h2) ** 2 / denom))

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _mapa_z(self, celdas: np.ndarray,
                mu: np.ndarray | None = None,
                sigma: np.ndarray | None = None) -> np.ndarray:
        """
        z-score por celda = max sobre los 4 canales de |x-μ|/σ.
        Retorna (ny, nx) float32.
        """
        mu    = self._mu if mu is None else mu
        sigma = self._sigma if sigma is None else sigma
        return (np.abs(celdas - mu) / sigma).max(axis=-1).astype(np.float32)

    @staticmethod
    def _score_de_mapa(mapa_z: np.ndarray) -> float:
        """Media de las TOP_K celdas con mayor z (ver comentario de TOP_K)."""
        plano = mapa_z.flatten()
        k = min(TOP_K, plano.size)
        return float(np.sort(plano)[-k:].mean())

    # ── Presencia ────────────────────────────────────────────────────────────

    def verificar_presencia(self, imagen_bgr: np.ndarray) -> dict:
        """
        Pre-check barato: ¿la imagen se parece globalmente a la referencia?
        Retorna {"ok", "razon", "detalle"}.  razon="sin_etiqueta" al fallar.
        """
        if not self._firma:
            return {"ok": True, "razon": None, "detalle": "sin firma"}

        firma_q = self._firma_imagen(imagen_bgr)

        brillo_mu  = self._firma["brillo_media"]
        brillo_sd  = max(self._firma["brillo_std"], 2.0)
        if abs(firma_q["brillo"] - brillo_mu) > BRILLO_SIGMAS * brillo_sd:
            return {
                "ok": False, "razon": "sin_etiqueta",
                "detalle": (f"Brillo {firma_q['brillo']:.0f} fuera del rango "
                            f"aprendido ({brillo_mu:.0f} ± "
                            f"{BRILLO_SIGMAS:.0f}·{brillo_sd:.1f})."),
            }

        dist = self._dist_chi2(firma_q["hist"], self._firma["hist_media"])
        tope = self._firma["hist_dist_max"] * HIST_DIST_FACTOR_MAX
        if dist > tope:
            return {
                "ok": False, "razon": "sin_etiqueta",
                "detalle": (f"Paleta de colores distinta a la referencia "
                            f"(χ²={dist:.3f} > {tope:.3f})."),
            }
        return {"ok": True, "razon": None, "detalle": "presencia OK"}

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def entrenar(self, rutas: list[str], callback_progreso=None) -> bool:
        """
        Aprende μ/σ por celda y calibra el umbral con leave-one-out.
        Rápido (segundos): apto para correr en la Jetson tras capturar el
        banco, sin GPU y sin liberar nada antes.

        Parámetros:
            rutas: imágenes del banco (rectificadas al canvas canónico, o
                crudas — `rectificar_si_necesario` decide).
            callback_progreso: callable(int 0-100, str) opcional para la UI.
        """
        def _progreso(pct, msg):
            if callback_progreso:
                callback_progreso(int(pct), msg)

        feats, firmas = [], []
        for i, ruta in enumerate(rutas):
            _progreso(90 * i / max(len(rutas), 1),
                      f"Analizando {os.path.basename(ruta)}")
            img = imread_safe(ruta)
            if img is None:
                logging.warning("DetectorROI.entrenar: no se pudo leer %s", ruta)
                continue
            rect, razon = rectificar_si_necesario(img)
            if rect is None:
                logging.warning("DetectorROI.entrenar: %s rechazada (%s)",
                                ruta, razon)
                continue
            feats.append(self._features_celdas(rect))
            firmas.append(self._firma_imagen(rect))

        n = len(feats)
        if n == 0:
            logging.error("DetectorROI.entrenar: ninguna imagen utilizable.")
            return False

        X = np.stack(feats, axis=0)                    # (n, ny, nx, 4)
        _progreso(92, "Calculando estadísticas por celda")

        self._mu    = X.mean(axis=0)
        self._sigma = np.maximum(X.std(axis=0), SIGMA_MIN)

        # Firma de presencia.
        hists   = np.stack([f["hist"] for f in firmas], axis=0)
        brillos = np.array([f["brillo"] for f in firmas], dtype=np.float32)
        hist_media = hists.mean(axis=0)
        self._firma = {
            "hist_media":    hist_media,
            "hist_dist_max": max(PISO_DIST_HIST,
                                 max(self._dist_chi2(h, hist_media)
                                     for h in hists)),
            "brillo_media":  float(brillos.mean()),
            "brillo_std":    float(brillos.std()),
        }

        # Calibración del umbral.
        _progreso(96, "Calibrando umbral (leave-one-out)")
        if n >= N_MIN_LOO:
            scores = self._scores_loo(X)
            q1, mediana, q3 = np.percentile(scores, [25, 50, 75])
            self._umbral = float(mediana + IQR_MULTIPLICADOR * (q3 - q1))
        else:
            # Con 1-2 imágenes el LOO no es viable: umbral provisional
            # holgado.  El operador debe capturar más buenas y reentrenar.
            score_max = max(
                self._score_de_mapa(self._mapa_z(X[i])) for i in range(n)
            ) or 1.0
            self._umbral = float(score_max * FACTOR_FALLBACK)
            logging.warning(
                "DetectorROI.entrenar: solo %d imágenes — umbral provisional "
                "%.3f (capturar ≥%d y reentrenar).", n, self._umbral, N_MIN_LOO
            )

        self._n_imagenes = n
        _progreso(100, "Entrenamiento completado")
        logging.info(
            "DetectorROI entrenado: %d imágenes, grilla %dx%d, umbral=%.3f",
            n, self._mu.shape[1], self._mu.shape[0], self._umbral,
        )
        return True

    def _scores_loo(self, X: np.ndarray) -> np.ndarray:
        """
        Score de cada imagen del banco contra las estadísticas de las otras
        n-1 (leave-one-out), vectorizado con suma y suma de cuadrados:
            μ_loo   = (Σx − x_i) / (n−1)
            var_loo = (Σx² − x_i²)/(n−1) − μ_loo²
        """
        n = X.shape[0]
        S  = X.sum(axis=0)
        S2 = (X ** 2).sum(axis=0)
        scores = np.empty(n, dtype=np.float32)
        for i in range(n):
            mu_i  = (S - X[i]) / (n - 1)
            var_i = np.maximum((S2 - X[i] ** 2) / (n - 1) - mu_i ** 2, 0.0)
            sd_i  = np.maximum(np.sqrt(var_i), SIGMA_MIN)
            scores[i] = self._score_de_mapa(self._mapa_z(X[i], mu_i, sd_i))
        return scores

    # ── Inspección ───────────────────────────────────────────────────────────

    def inspeccionar(self, imagen_bgr: np.ndarray,
                     umbral: float | None = None) -> dict:
        """
        Evalúa una imagen.  Orden: rectificación → presencia → celdas.

        Retorna dict con las mismas claves que el detector original:
            {ok, score, umbral, mapa_anomalia, imagen_rectificada,
             disponible, defecto, detalle}
        más "celdas_malas": [{"bbox": [x,y,w,h], "roi": str, "z": float}]
        (celdas sobre el umbral, para pintarlas en la UI).
        defecto ∈ {None, "sin_etiqueta", "color"}.
        """
        if not self.entrenado:
            logging.warning("DetectorROI.inspeccionar(): modelo no entrenado.")
            return {"ok": True, "score": 0.0, "umbral": 0.0,
                    "mapa_anomalia": None, "imagen_rectificada": None,
                    "disponible": False, "defecto": None, "detalle": "",
                    "celdas_malas": None}

        img_rect, razon = rectificar_si_necesario(imagen_bgr)
        if img_rect is None:
            detalles = {
                "sin_segmentacion":  "No se pudo segmentar la etiqueta "
                                     "(fondo o luz atípica).",
                "contorno_vacio":    "La máscara de la etiqueta no dio contornos.",
                "contorno_diminuto": "Etiqueta demasiado pequeña en el frame.",
                "etiqueta_cortada":  "Etiqueta fuera del área de inspección — "
                                     "debe estar completa dentro del domo.",
            }
            return {
                "ok": False, "score": None,
                "umbral": float(umbral if umbral is not None else self._umbral),
                "mapa_anomalia": None, "imagen_rectificada": None,
                "disponible": True, "defecto": "sin_etiqueta",
                "detalle": detalles.get(razon, f"Rectificación falló ({razon})."),
                "celdas_malas": None,
            }

        presencia = self.verificar_presencia(img_rect)
        if not presencia["ok"]:
            return {
                "ok": False, "score": None,
                "umbral": float(umbral if umbral is not None else self._umbral),
                "mapa_anomalia": None, "imagen_rectificada": img_rect,
                "disponible": True, "defecto": "sin_etiqueta",
                "detalle": presencia["detalle"],
                "celdas_malas": None,
            }

        celdas = self._features_celdas(img_rect)
        mapa_z = self._mapa_z(celdas)
        score  = self._score_de_mapa(mapa_z)
        thr    = float(umbral if umbral is not None else self._umbral)
        ok     = score < thr

        # Mapa de anomalía a resolución del canvas (diagnóstico; la UI dibuja
        # las celdas_malas como rectángulos sobre la rectificada).
        mapa_full = cv2.resize(mapa_z, (img_rect.shape[1], img_rect.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        celdas_malas = []
        if not ok:
            ys, xs = np.where(mapa_z >= thr)
            # Ordenadas por z descendente; tope de 20 para no inundar la UI.
            orden = np.argsort(-mapa_z[ys, xs])[:20]
            for idx in orden:
                cy, cx = int(ys[idx]), int(xs[idx])
                bbox = [cx * CELDA_PX, cy * CELDA_PX, CELDA_PX, CELDA_PX]
                celdas_malas.append({
                    "bbox": bbox,
                    "roi":  self._nombre_roi(bbox),
                    "z":    float(mapa_z[cy, cx]),
                })

        detalle = "Colores dentro de lo aprendido."
        if not ok:
            zonas = sorted({c["roi"] for c in celdas_malas})
            detalle = (f"Color fuera de rango en {len(celdas_malas)} celda(s) "
                       f"— zonas: {', '.join(zonas)} "
                       f"(score={score:.2f} / umbral={thr:.2f}).")

        return {
            "ok": ok,
            "score": round(score, 4),
            "umbral": round(thr, 4),
            "mapa_anomalia": mapa_full,
            "imagen_rectificada": img_rect,
            "disponible": True,
            "defecto": None if ok else "color",
            "detalle": detalle,
            "celdas_malas": celdas_malas,
        }

    def _nombre_roi(self, bbox_celda: list) -> str:
        """Nombre de la ROI semántica que contiene el centro de la celda."""
        cx = bbox_celda[0] + bbox_celda[2] // 2
        cy = bbox_celda[1] + bbox_celda[3] // 2
        for roi in self._rois:
            x, y, w, h = roi.get("bbox", (0, 0, 0, 0))
            if x <= cx < x + w and y <= cy < y + h:
                return roi.get("nombre", "zona")
        return "zona sin nombre"

    # ── Persistencia ─────────────────────────────────────────────────────────

    def guardar(self, ruta: str) -> bool:
        if not self.entrenado:
            logging.error("DetectorROI.guardar(): modelo no entrenado.")
            return False
        try:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            ruta_tmp = ruta + ".tmp"
            with open(ruta_tmp, "wb") as f:
                pickle.dump({
                    "version":    PKL_VERSION,
                    "canvas":     [CANONICAL_W, CANONICAL_H],
                    "celda":      CELDA_PX,
                    "mu":         self._mu,
                    "sigma":      self._sigma,
                    "umbral":     self._umbral,
                    "firma":      self._firma,
                    "n_imagenes": self._n_imagenes,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(ruta_tmp, ruta)
            logging.info("DetectorROI: modelo guardado en '%s'", ruta)
            return True
        except Exception as e:
            logging.error("DetectorROI.guardar: %s", e)
            return False

    def cargar(self, ruta: str) -> bool:
        if not os.path.exists(ruta):
            logging.info("DetectorROI.cargar: '%s' no existe aún.", ruta)
            return False
        try:
            with open(ruta, "rb") as f:
                datos = pickle.load(f)

            if datos.get("version", 0) != PKL_VERSION:
                logging.error(
                    "DetectorROI.cargar: versión v%s ≠ v%d del código. "
                    "Reentrenar.", datos.get("version"), PKL_VERSION,
                )
                return False
            if tuple(datos.get("canvas", ())) != (CANONICAL_W, CANONICAL_H):
                logging.error(
                    "DetectorROI.cargar: canvas del modelo %s ≠ %s del código. "
                    "Reentrenar.", datos.get("canvas"),
                    (CANONICAL_W, CANONICAL_H),
                )
                return False
            if datos.get("celda") != CELDA_PX:
                logging.error(
                    "DetectorROI.cargar: celda del modelo %s ≠ %d del código. "
                    "Reentrenar.", datos.get("celda"), CELDA_PX,
                )
                return False

            self._mu         = datos["mu"]
            self._sigma      = datos["sigma"]
            self._umbral     = datos["umbral"]
            self._firma      = datos["firma"]
            self._n_imagenes = datos.get("n_imagenes", 0)
            logging.info(
                "DetectorROI: modelo cargado de '%s' — grilla %dx%d, "
                "umbral=%.3f, n=%d",
                ruta, self._mu.shape[1], self._mu.shape[0],
                self._umbral, self._n_imagenes,
            )
            return True
        except Exception as e:
            logging.error("DetectorROI.cargar: %s", e)
            return False

