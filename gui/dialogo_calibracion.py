"""
DialogoCalibracion - Calibrar una referencia (auto + edición manual)
======================================================================

Flujo del diálogo:

    1. **Color (auto + edición)**: al abrirse, k-means HSV detecta la
       paleta y popula la tabla de ROIs.  El operador puede:
         - Renombrar cualquier ROI (editar inline en la tabla).
         - Eliminar un ROI con "− ROI".
         - Agregar uno nuevo con "+ ROI" (cv2.selectROI sobre la maestra
           → pedir nombre → extraer rangos HSV automáticamente).
         - Re-pintar el bbox de uno existente con "Editar bbox" (recalcula
           rangos sobre la nueva región).
         - Pulsar "Volver a detectar paleta" para regenerar todo desde
           cero (descarta cualquier edición).

    2. **Persistencia**: al confirmar guarda
           referencias/<ref>/calibracion_color.json
           referencias/<ref>/plantilla_maestra.jpg
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QInputDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QProgressDialog,
)

from logica import calibracion_referencia
from logica.cv_util import imwrite_safe


# Paleta de colores BGR para overlay de cada ROI en el preview.  Hasta 8.
_COLORES_OVERLAY = [
    (0, 255, 255),   # amarillo
    (255, 0, 255),   # magenta
    (0, 255, 0),     # verde
    (0, 128, 255),   # naranja
    (255, 255, 0),   # cian
    (128, 0, 255),   # violeta
    (255, 128, 0),   # azul claro
    (0, 0, 255),     # rojo
]


# ── Worker thread para no bloquear el UI durante k-means ──────────────────

class _HiloKMeans(QThread):
    """Corre `calibrar_automaticamente` en background.

    K-means con ~80k muestras suele tardar 2-4 s en Jetson Orin Nano
    pero puede subir si la memoria está apretada — sin un thread aparte
    el UI se congela y parece que la app crasheó.
    """
    terminado = pyqtSignal(object, str)  # (rois | None, error_msg)

    def __init__(self, maestra: np.ndarray):
        super().__init__()
        self._maestra = maestra

    def run(self):
        try:
            rois = calibracion_referencia.calibrar_automaticamente(self._maestra)
            self.terminado.emit(rois, "")
        except Exception as e:
            logging.exception("Error en calibrar_automaticamente")
            self.terminado.emit(None, str(e))


class DialogoCalibracion(QDialog):
    """Diálogo modal para calibrar la paleta de color (auto + edición)."""

    def __init__(
        self,
        parent,
        nombre_referencia: str,
        carpeta_referencia: str,
        imagen_maestra_rect_bgr: np.ndarray,
    ):
        super().__init__(parent)
        self._nombre  = nombre_referencia
        self._carpeta = Path(carpeta_referencia)
        self._maestra = imagen_maestra_rect_bgr.copy()
        # Lista mutable de ROIs.  Cada item: dict {"nombre", "bbox",
        # "rangos", "stats"} — mismo formato que `calibrar_automaticamente`.
        self._rois: list[dict] = []
        self._suspender_signals = False  # evita loops cuando rellenamos tabla

        self.setWindowTitle(f"Calibrar referencia — {nombre_referencia}")
        self.setMinimumSize(1000, 720)
        self._crear_interfaz()

        # Auto-detectar al abrir.
        self._detectar_paleta()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _crear_interfaz(self):
        layout = QVBoxLayout(self)

        cabecera = QLabel(
            f"<b>Calibración de '{self._nombre}'</b><br>"
            "El sistema detecta la paleta automáticamente. Puedes editarla, "
            "agregar o quitar ROIs antes de guardar."
        )
        cabecera.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(cabecera)

        contenido = QHBoxLayout()
        layout.addLayout(contenido, stretch=1)

        # Preview de la maestra con overlays de ROIs.
        self.lbl_maestra = QLabel()
        self.lbl_maestra.setMinimumSize(QSize(560, 320))
        self.lbl_maestra.setAlignment(Qt.AlignCenter)
        self.lbl_maestra.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #333; border-radius: 6px;"
        )
        contenido.addWidget(self.lbl_maestra, stretch=1)

        # Panel derecho.
        panel = QVBoxLayout()
        panel.setSpacing(10)
        contenido.addLayout(panel)

        # ── ROIs (tabla editable) ─────────────────────────────────────────────
        grp_color = QGroupBox("Paleta de color (ROIs)")
        col1 = QVBoxLayout(grp_color)
        col1.setSpacing(6)

        self.tbl_rois = QTableWidget(0, 4)
        self.tbl_rois.setHorizontalHeaderLabels(["", "Nombre", "Bbox (x,y,w,h)", "% label"])
        self.tbl_rois.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.tbl_rois.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.tbl_rois.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.tbl_rois.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.tbl_rois.setMinimumHeight(180)
        self.tbl_rois.itemChanged.connect(self._on_celda_rois_cambiada)
        col1.addWidget(self.tbl_rois)

        botones_rois = QHBoxLayout()
        self.btn_add_roi  = QPushButton("+ ROI")
        self.btn_add_roi.clicked.connect(self._agregar_roi_manual)
        self.btn_edit_roi = QPushButton("Editar bbox")
        self.btn_edit_roi.clicked.connect(self._editar_bbox_roi)
        self.btn_del_roi  = QPushButton("− ROI")
        self.btn_del_roi.clicked.connect(self._quitar_roi)
        botones_rois.addWidget(self.btn_add_roi)
        botones_rois.addWidget(self.btn_edit_roi)
        botones_rois.addWidget(self.btn_del_roi)
        col1.addLayout(botones_rois)

        self.btn_recalibrar = QPushButton("Volver a detectar paleta (descarta ediciones)")
        self.btn_recalibrar.setMinimumHeight(34)
        self.btn_recalibrar.setStyleSheet(
            "QPushButton { background-color: #455A64; color: white;"
            " border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:pressed { background-color: #263238; }"
        )
        self.btn_recalibrar.clicked.connect(self._detectar_paleta)
        col1.addWidget(self.btn_recalibrar)

        panel.addWidget(grp_color, stretch=1)

        # ── Botones inferiores ────────────────────────────────────────────────
        fila_final = QHBoxLayout()
        fila_final.addStretch()
        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_cancelar.clicked.connect(self.reject)
        self.btn_guardar  = QPushButton("Guardar y cerrar")
        self.btn_guardar.setDefault(True)
        self.btn_guardar.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white;"
            " font-weight: bold; border-radius: 6px; padding: 8px 18px; }"
            "QPushButton:pressed { background-color: #1B5E20; }"
        )
        self.btn_guardar.clicked.connect(self._guardar)
        fila_final.addWidget(self.btn_cancelar)
        fila_final.addWidget(self.btn_guardar)
        layout.addLayout(fila_final)

    # ── Detección automática inicial ──────────────────────────────────────────

    def _detectar_paleta(self):
        """K-means en hilo separado para no congelar el UI."""
        self._prog_kmeans = QProgressDialog(
            "Detectando paleta con k-means HSV (puede tardar 2-5 s)…",
            None, 0, 0, self,
        )
        self._prog_kmeans.setWindowTitle("Calibrando")
        self._prog_kmeans.setWindowModality(Qt.WindowModal)
        self._prog_kmeans.setCancelButton(None)
        self._prog_kmeans.setAutoClose(False)
        self._prog_kmeans.show()

        # Deshabilitar acciones que podrían chocar mientras corre.
        self.btn_recalibrar.setEnabled(False)
        self.btn_add_roi.setEnabled(False)

        self._hilo_kmeans = _HiloKMeans(self._maestra)
        self._hilo_kmeans.terminado.connect(self._kmeans_terminado)
        self._hilo_kmeans.start()

    def _kmeans_terminado(self, rois, error_msg: str):
        if hasattr(self, "_prog_kmeans") and self._prog_kmeans is not None:
            self._prog_kmeans.close()
            self._prog_kmeans = None
        self.btn_recalibrar.setEnabled(True)
        self.btn_add_roi.setEnabled(True)

        if rois is None:
            QMessageBox.warning(self, "No se pudo calibrar",
                                f"K-means falló:\n{error_msg}")
            self._rois = []
        else:
            self._rois = rois
        logging.info("DialogoCalibracion: detectados %d ROIs auto", len(self._rois))
        self._refrescar_tabla_rois()
        self._refrescar_preview()

    # ── Tabla de ROIs ─────────────────────────────────────────────────────────

    def _refrescar_tabla_rois(self):
        """Vuelca self._rois a la tabla.  Bloquea signals para no recursar."""
        self._suspender_signals = True
        try:
            self.tbl_rois.setRowCount(0)
            for i, roi in enumerate(self._rois):
                self.tbl_rois.insertRow(i)

                # Cuadrito de color.
                color_bgr = _COLORES_OVERLAY[i % len(_COLORES_OVERLAY)]
                qcolor = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
                celda_color = QTableWidgetItem(" ")
                celda_color.setBackground(qcolor)
                celda_color.setFlags(celda_color.flags() & ~Qt.ItemIsEditable)
                self.tbl_rois.setItem(i, 0, celda_color)

                # Nombre (editable).
                celda_nombre = QTableWidgetItem(roi["nombre"])
                self.tbl_rois.setItem(i, 1, celda_nombre)

                # Bbox (read-only).
                x, y, w, h = roi["bbox"]
                celda_bbox = QTableWidgetItem(f"{x},{y},{w},{h}")
                celda_bbox.setFlags(celda_bbox.flags() & ~Qt.ItemIsEditable)
                self.tbl_rois.setItem(i, 2, celda_bbox)

                # Fracción (read-only).
                frac = roi["stats"].get("fraccion_label", 0.0) * 100
                celda_frac = QTableWidgetItem(f"{frac:.0f}%")
                celda_frac.setFlags(celda_frac.flags() & ~Qt.ItemIsEditable)
                celda_frac.setTextAlignment(Qt.AlignCenter)
                self.tbl_rois.setItem(i, 3, celda_frac)
        finally:
            self._suspender_signals = False

    def _on_celda_rois_cambiada(self, item):
        """El operador editó una celda de la tabla — sincroniza con self._rois."""
        if self._suspender_signals:
            return
        fila = item.row()
        col  = item.column()
        if fila >= len(self._rois):
            return
        if col == 1:  # nombre
            nuevo = item.text().strip()
            if not nuevo:
                # No permitir nombre vacío — restaurar el anterior.
                self._suspender_signals = True
                try:
                    item.setText(self._rois[fila]["nombre"])
                finally:
                    self._suspender_signals = False
                return
            self._rois[fila]["nombre"] = nuevo
            logging.info("ROI %d renombrado a '%s'", fila, nuevo)

    def _agregar_roi_manual(self):
        """Pinta un nuevo ROI sobre la maestra y calcula rangos automáticamente."""
        nombre, ok = QInputDialog.getText(
            self, "Nombre del ROI nuevo",
            "Nombre descriptivo (banner_verde, logo_azul, …):",
            text=f"roi_{len(self._rois) + 1}",
        )
        if not ok or not nombre.strip():
            return
        nombre = nombre.strip()

        QMessageBox.information(
            self, "Pintar ROI",
            "En la ventana que abriremos a continuación:\n"
            "  • Arrastra para definir el rectángulo sobre el COLOR objetivo.\n"
            "  • ENTER o ESPACIO para aceptar.\n"
            "  • C para cancelar."
        )

        bbox = self._pedir_bbox_a_operador(f"Pintar ROI: {nombre}")
        if bbox is None:
            return

        try:
            extracto = calibracion_referencia.extraer_rango_hsv(self._maestra, bbox)
        except ValueError as e:
            QMessageBox.warning(self, "ROI inválido", str(e))
            return

        # Calcular fracción del label que ocupa este bbox (aproximación
        # área del bbox / área del canvas — no es la fracción real del
        # cluster pero da idea para mostrar en la tabla).
        x, y, w, h = bbox
        frac = (w * h) / (self._maestra.shape[0] * self._maestra.shape[1])

        self._rois.append({
            "nombre": nombre,
            "bbox":   list(bbox),
            "rangos": extracto["rangos"],
            "stats": {
                **extracto["stats"],
                "fraccion_label": frac,
            },
        })
        logging.info("ROI '%s' agregado manualmente: bbox=%s", nombre, bbox)
        self._refrescar_tabla_rois()
        self._refrescar_preview()

    def _editar_bbox_roi(self):
        """Re-pinta el bbox del ROI seleccionado y recalcula rangos."""
        fila = self.tbl_rois.currentRow()
        if fila < 0 or fila >= len(self._rois):
            QMessageBox.information(
                self, "Selecciona un ROI",
                "Marca primero un ROI en la tabla para editar su bbox.",
            )
            return

        roi = self._rois[fila]
        bbox = self._pedir_bbox_a_operador(
            f"Editar bbox de '{roi['nombre']}'"
        )
        if bbox is None:
            return

        try:
            extracto = calibracion_referencia.extraer_rango_hsv(self._maestra, bbox)
        except ValueError as e:
            QMessageBox.warning(self, "ROI inválido", str(e))
            return

        x, y, w, h = bbox
        frac = (w * h) / (self._maestra.shape[0] * self._maestra.shape[1])
        roi["bbox"]   = list(bbox)
        roi["rangos"] = extracto["rangos"]
        roi["stats"]  = {**extracto["stats"], "fraccion_label": frac}
        logging.info("ROI '%s' bbox actualizado: %s", roi["nombre"], bbox)
        self._refrescar_tabla_rois()
        self._refrescar_preview()

    def _quitar_roi(self):
        fila = self.tbl_rois.currentRow()
        if fila < 0 or fila >= len(self._rois):
            return
        nombre = self._rois[fila]["nombre"]
        del self._rois[fila]
        logging.info("ROI '%s' eliminado", nombre)
        self._refrescar_tabla_rois()
        self._refrescar_preview()

    def _pedir_bbox_a_operador(self, titulo_ventana: str) -> tuple | None:
        """
        Abre `cv2.selectROI` sobre una maestra reescalada para que entre en
        la pantalla.  Retorna (x, y, w, h) en coordenadas del canvas
        canónico, o None si el operador canceló.
        """
        preview, factor = self._maestra_para_seleccion()
        cv2.namedWindow(titulo_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(titulo_ventana, preview.shape[1], preview.shape[0])
        sel = cv2.selectROI(titulo_ventana, preview,
                            showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(titulo_ventana)
        x_p, y_p, w_p, h_p = (int(v) for v in sel)
        if w_p == 0 or h_p == 0:
            return None
        return (
            int(round(x_p / factor)),
            int(round(y_p / factor)),
            int(round(w_p / factor)),
            int(round(h_p / factor)),
        )

    def _maestra_para_seleccion(self) -> tuple[np.ndarray, float]:
        """Reescala la maestra para `cv2.selectROI` a un tamaño manejable."""
        max_w, max_h = 1600, 900
        h, w = self._maestra.shape[:2]
        factor = min(max_w / w, max_h / h, 1.0)
        if factor >= 0.999:
            return self._maestra.copy(), 1.0
        nw, nh = int(round(w * factor)), int(round(h * factor))
        return cv2.resize(self._maestra, (nw, nh),
                          interpolation=cv2.INTER_AREA), factor

    # ── Preview ───────────────────────────────────────────────────────────────

    def _refrescar_preview(self):
        """Pinta la maestra con bbox + nombre por ROI con su color asignado."""
        if self._maestra is None:
            return
        overlay = self._maestra.copy()
        for i, roi in enumerate(self._rois):
            color_bgr = _COLORES_OVERLAY[i % len(_COLORES_OVERLAY)]
            x, y, w, h = roi["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, 6)
            cv2.putText(
                overlay, roi["nombre"], (x + 8, max(y + 32, 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_bgr, 3, cv2.LINE_AA,
            )

        h_lbl = max(self.lbl_maestra.height(), 320)
        w_lbl = max(self.lbl_maestra.width(),  560)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            w_lbl, h_lbl, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_maestra.setPixmap(pix)

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _guardar(self):
        if not self._rois:
            QMessageBox.warning(
                self, "Sin ROIs definidos",
                "Agrega al menos un ROI antes de guardar.",
            )
            return
        # Validar nombres únicos.
        nombres = [r["nombre"] for r in self._rois]
        if len(set(nombres)) < len(nombres):
            duplicados = {n for n in nombres if nombres.count(n) > 1}
            QMessageBox.warning(
                self, "Nombres duplicados",
                f"Hay ROIs con el mismo nombre: {', '.join(sorted(duplicados))}.\n"
                "Renómbralos antes de guardar.",
            )
            return

        ok_color = calibracion_referencia.guardar_calibracion_color(
            self._carpeta, self._rois,
        )
        ok_master = imwrite_safe(
            str(calibracion_referencia.ruta_maestra(self._carpeta)),
            self._maestra,
        )

        if not (ok_color and ok_master):
            QMessageBox.critical(
                self, "Error al guardar",
                f"  calibracion_color.json: {'OK' if ok_color else 'FAIL'}\n"
                f"  plantilla_maestra.jpg:  {'OK' if ok_master else 'FAIL'}",
            )
            return

        QMessageBox.information(
            self, "Calibración guardada",
            f"Referencia '{self._nombre}' calibrada.\n\n"
            f"• {len(self._rois)} ROIs ({', '.join(nombres)}).\n\n"
            "Ya puedes capturar el banco y entrenar el modelo.",
        )
        self.accept()

    # ── Eventos ───────────────────────────────────────────────────────────────

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self._refrescar_preview()

    def closeEvent(self, evento):
        """Espera al hilo de k-means en curso antes de cerrar — evita crashes."""
        hilo = getattr(self, "_hilo_kmeans", None)
        if hilo is not None and hilo.isRunning():
            logging.info("Esperando que termine _hilo_kmeans antes de cerrar…")
            hilo.wait(60_000)  # hasta 60 s
        super().closeEvent(evento)
