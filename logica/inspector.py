"""
Inspector - Orquestador del pipeline de análisis (variante ROI)
================================================================
Coordina el detector de manchones/color por celdas (DetectorROI) y
devuelve un resultado único por etiqueta.

Esta variante NO tiene PatchCore ni OCR (ver CLAUDE.md): detecta cambios
de color, manchas y oclusiones comparando estadísticas HSV por celda
contra un banco de etiquetas buenas.  Todo en CPU, sin GPU.

Este objeto no conoce qué referencia está activa — solo recibe la ruta
del modelo y un umbral override.  MainWindow se encarga de pasarle los
valores correctos cuando el usuario cambia de referencia.

Defensa en capas contra falsos positivos (domo vacío / luz apagada):
    1. Pre-check de brillo muy oscuro → "sin_luz" (barato, sin modelo).
    2. Presencia dentro del detector (hist HSV global + brillo) →
       "sin_etiqueta" si la imagen no coincide con la paleta del banco.
    3. Celdas → "color" si alguna zona desvía de lo aprendido.
"""

import logging
import numpy as np

from logica.detector_roi import DetectorROI

# Brillo mínimo aceptable (valor medio de píxel en escala 0-255).  Por debajo
# se asume luz apagada/imagen negra y se rechaza antes de correr cualquier
# comparación.
BRILLO_MINIMO = 30


class Inspector:
    """
    Orquesta la inspección de una etiqueta con el detector por celdas.

    Uso:
        inspector = Inspector()
        inspector.preparar(
            ruta_modelo="referencias/Lannate_TM_SL/modelo_roi.pkl",
            umbral_override=None,
            rois=[...],          # ROIs de calibracion_color.json (nombres)
        )
        resultado = inspector.inspeccionar(imagen_bgr)
        # resultado = {
        #     "ok":            True/False,
        #     "defecto":       None | "sin_luz" | "sin_etiqueta" | "color",
        #     "detalle":       "...",
        #     "scores":        {"color": float | None},
        #     "umbral":        float | None,
        #     "mapa_anomalia": np.ndarray | None,
        #     "celdas_malas":  list | None,
        # }
    """

    def __init__(self):
        self._detector           = DetectorROI()
        self._ruta_modelo_actual = None
        self._listo              = False

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    def preparar(self, ruta_modelo: str = None,
                       umbral_override: float = None,
                       rangos_hsv=None,
                       rois: list | None = None) -> bool:
        """
        Carga el modelo desde disco y aplica el umbral override si se pasó.

        Parámetros:
            ruta_modelo (str): Ruta al .pkl. None = no cargar nada.
            umbral_override (float): Umbral que reemplaza al calculado al
                entrenar. None = usar el del modelo.
            rangos_hsv: aceptado por compatibilidad de firma con los callers
                del proyecto original — la segmentación siempre usa NO_DOME
                (ver "trampa del trapecio" en cv_util), así que se ignora.
            rois: ROIs semánticas de `calibracion_color.json` — solo para
                nombrar zonas en los mensajes de defecto.

        Retorna:
            True si se cargó un modelo listo para inspeccionar.
        """
        self._ruta_modelo_actual = ruta_modelo

        # Detector nuevo cada vez — así al cambiar de referencia no heredamos
        # las estadísticas de la referencia anterior.
        self._detector = DetectorROI()
        self._detector.set_calibracion(rois)

        if not ruta_modelo:
            logging.info(
                "Inspector.preparar(): sin ruta de modelo → inspección deshabilitada."
            )
            self._listo = True
            return False

        exito = self._detector.cargar(ruta_modelo)

        if exito and umbral_override is not None:
            self._detector.umbral = float(umbral_override)
            logging.info(
                f"Inspector.preparar(): umbral override = {umbral_override:.4f}"
            )
        elif not exito:
            logging.warning(
                f"Inspector.preparar(): modelo no encontrado en '{ruta_modelo}'."
            )

        self._listo = True
        return exito

    @property
    def esta_listo(self) -> bool:
        return self._listo

    @property
    def tiene_modelo(self) -> bool:
        """True si hay un modelo cargado y listo para inspeccionar."""
        return self._detector.entrenado

    @property
    def detector(self) -> DetectorROI:
        """Acceso directo al detector (para ajustar umbral desde la UI)."""
        return self._detector

    @property
    def ruta_modelo_actual(self):
        return self._ruta_modelo_actual

    # ── Inspección producción ──────────────────────────────────────────────────

    def inspeccionar(self, imagen_bgr: np.ndarray) -> dict:
        """
        Evalúa la imagen y clasifica la etiqueta como OK o defectuosa.

        El dict devuelto incluye siempre:
            - ok, defecto, detalle
            - scores: {"color": float | None}
            - umbral: float | None
            - mapa_anomalia / imagen_rectificada / celdas_malas
        """
        umbral_actual = (self._detector.umbral
                         if self._detector.entrenado else None)

        if imagen_bgr.mean() < BRILLO_MINIMO:
            return {
                "ok":                 False,
                "defecto":            "sin_luz",
                "detalle":            "Imagen demasiado oscura — luz apagada o sin etiqueta.",
                "scores":             {"color": None},
                "umbral":             umbral_actual,
                "mapa_anomalia":      None,
                "imagen_rectificada": None,
                "celdas_malas":       None,
            }

        res = self._detector.inspeccionar(imagen_bgr)

        # Sin modelo cargado → no podemos decidir.
        if not res["disponible"]:
            return {
                "ok":                 True,
                "defecto":            None,
                "detalle":            "Sin modelo entrenado — inspección omitida.",
                "scores":             {"color": None},
                "umbral":             None,
                "mapa_anomalia":      None,
                "imagen_rectificada": None,
                "celdas_malas":       None,
            }

        return {
            "ok":                 res["ok"],
            "defecto":            res.get("defecto"),
            "detalle":            res.get("detalle", ""),
            "scores":             {"color": res.get("score")},
            "umbral":             res.get("umbral", umbral_actual),
            "mapa_anomalia":      res.get("mapa_anomalia"),
            "imagen_rectificada": res.get("imagen_rectificada"),
            "celdas_malas":       res.get("celdas_malas"),
        }

    # ── Inspección diagnóstico (siempre retorna el mapa completo) ──────────────

    def inspeccionar_diagnostico(self, imagen_bgr: np.ndarray) -> dict:
        """
        Igual que inspeccionar(), pero pensado para la pantalla de
        mantenimiento: siempre retorna el mapa de anomalías completo para
        visualización, aún cuando la etiqueta esté OK.
        """
        if imagen_bgr.mean() < BRILLO_MINIMO:
            return {
                "ok":                 False,
                "disponible":         True,
                "score":              None,
                "umbral":             0.0,
                "mapa_anomalia":      None,
                "imagen_rectificada": None,
                "defecto":            "sin_luz",
                "detalle":            "Imagen demasiado oscura — luz apagada o sin etiqueta.",
                "celdas_malas":       None,
            }

        res = self._detector.inspeccionar(imagen_bgr)
        return {
            "ok":                 res["ok"],
            "disponible":         res["disponible"],
            "score":              res.get("score"),
            "umbral":             res.get("umbral", 0.0),
            "mapa_anomalia":      res.get("mapa_anomalia"),
            "imagen_rectificada": res.get("imagen_rectificada"),
            "defecto":            res.get("defecto"),
            "detalle":            res.get("detalle", ""),
            "celdas_malas":       res.get("celdas_malas"),
        }
