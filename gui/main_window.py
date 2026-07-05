"""
MainWindow - La ventana principal del prototipo
=================================================
Marco con QStackedWidget que muestra una vista a la vez:
    - Operación     (pantalla de producción + captura-y-analiza manual)
    - Manual        (captura suelta de imágenes)
    - Mantenimiento (calibración, banco, entrenamiento)
    - Referencias   (elegir/crear/borrar referencias de etiqueta)

MainWindow es el orquestador:
    • Coordina Configuracion (config global) con GestorReferencias (por ref).
    • Al cambiar de referencia, recarga Inspector con el .pkl correcto.
    • El recorte de cámara es GLOBAL (vive en calibracion.json) — todas las
      referencias lo comparten.  Antes era por-referencia en meta.json; ver
      `_migrar_recorte_global_si_aplica` para la migración silenciosa.
    • El umbral del modelo sí es por referencia (en meta.json).

Prototipo: no hay motor, sensor IR ni banda — la captura es manual desde
el botón de Operación. El hardware automático se agrega cuando el cliente
acepte la propuesta y se pase al diseño final.
"""

import os
import re
import logging
import cv2

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QImage, QPixmap

from gui.vista_operacion      import VistaOperacion
from gui.vista_manual         import VistaManual
from gui.vista_mantenimiento  import VistaMantenimiento
from gui.vista_referencias    import VistaReferencias
from gui.dialogo_calibracion  import DialogoCalibracion

from hardware.camara           import Camara
from logica.configuracion      import Configuracion
from logica.cv_util            import imwrite_safe, rectificar_etiqueta, RANGOS_HSV_NO_DOME
from logica.inspector          import Inspector
from logica.gestor_referencias import GestorReferencias
from logica.detector_roi      import DetectorROI
from logica                    import calibracion_referencia


# Ruta base del proyecto (para fotos_prueba, que es global — no por referencia)
_RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Worker thread para entrenamiento (no bloquea la UI) ───────────────────────

class _HiloEntrenamiento(QThread):
    """
    Ejecuta DetectorROI.entrenar() en un hilo secundario.

    El entrenamiento por celdas tarda <1 s, pero se mantiene el QThread
    para no congelar la UI si el banco crece o el disco (Google Drive)
    responde lento al leer las imágenes.
    """

    progreso  = pyqtSignal(int, str)   # (porcentaje 0-100, mensaje)
    terminado = pyqtSignal(bool, str)  # (exito, mensaje)

    def __init__(self, detector: DetectorROI, rutas: list,
                 ruta_salida_pkl: str):
        super().__init__()
        self._detector = detector
        self._rutas    = rutas
        self._ruta_pkl = ruta_salida_pkl

    def run(self):
        try:
            exito = self._detector.entrenar(
                self._rutas,
                callback_progreso=lambda pct, msg: self.progreso.emit(pct, msg)
            )
            if exito:
                self._detector.guardar(self._ruta_pkl)
                self.terminado.emit(True, "Modelo entrenado y guardado correctamente.")
            else:
                self.terminado.emit(False,
                    "Entrenamiento fallido: revisa las imágenes del banco.")
        except Exception as e:
            logging.exception("Entrenamiento del modelo de color falló")
            self.terminado.emit(False, f"Error durante el entrenamiento:\n{e}")


# ── Ventana principal ──────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """La ventana principal — orquestador de vistas y referencias."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Inspector de Etiquetas Individuales")
        self.setMinimumSize(1024, 600)

        self._config = Configuracion()
        self._config.cargar()

        self._gestor = GestorReferencias()

        self._camara = Camara(self._config.config_camara)
        self._camara.abrir()

        self._ultimo_frame       = None
        self._hilo_entrenamiento = None
        self._dialogo_progreso   = None

        # Calibración de la referencia activa: rangos HSV (se conservan por
        # compatibilidad de firma — la segmentación siempre usa NO_DOME) y
        # ROIs semánticas para nombrar las zonas defectuosas en los mensajes
        # ("mancha en banner verde").  Se cargan en `_aplicar_referencia` y
        # se propagan a Inspector y al detector nuevo en cada entrenamiento.
        self._rangos_hsv_activos: list | None = None
        self._rois_activas: list | None = None

        self._timer_camara_viva  = QTimer(self)
        self._timer_camara_viva.setInterval(66)   # ≈ 15 fps
        self._timer_camara_viva.timeout.connect(self._tick_camara_viva)

        # Inspector sin modelo cargado — se prepara cuando se activa una referencia.
        self._inspector = Inspector()

        self._crear_interfaz()
        self._conectar_senales()

        # Cargar la referencia activa (si hay). Esto es lo último — deja la UI
        # en el estado correcto según qué haya en disco.
        self._cargar_referencia_inicial()

    # ── Construcción de la interfaz ────────────────────────────────────────────

    def _crear_interfaz(self):
        contenedor_central = QWidget()
        self.setCentralWidget(contenedor_central)

        layout_principal = QVBoxLayout()
        contenedor_central.setLayout(layout_principal)

        self.vista_operacion     = VistaOperacion()
        self.vista_manual        = VistaManual()
        self.vista_mantenimiento = VistaMantenimiento()
        self.vista_referencias   = VistaReferencias()

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.vista_operacion)      # índice 0
        self.stacked.addWidget(self.vista_manual)          # índice 1
        self.stacked.addWidget(self.vista_mantenimiento)   # índice 2
        self.stacked.addWidget(self.vista_referencias)     # índice 3
        self.stacked.setCurrentIndex(0)

        barra_navegacion = QHBoxLayout()

        self.btn_operacion     = QPushButton("Operación")
        self.btn_manual        = QPushButton("Manual")
        self.btn_mantenimiento = QPushButton("Mantenimiento")
        self.btn_referencias   = QPushButton("Referencias")

        for btn in [self.btn_operacion, self.btn_manual,
                    self.btn_mantenimiento, self.btn_referencias]:
            btn.setMinimumHeight(50)

        barra_navegacion.addWidget(self.btn_operacion)
        barra_navegacion.addWidget(self.btn_manual)
        barra_navegacion.addWidget(self.btn_mantenimiento)
        barra_navegacion.addWidget(self.btn_referencias)

        # Tuple alineado al orden del QStackedWidget (0=operación, 1=manual, ...).
        self._botones_nav = (
            self.btn_operacion,
            self.btn_manual,
            self.btn_mantenimiento,
            self.btn_referencias,
        )
        self._actualizar_botones_navegacion(0)

        layout_principal.addWidget(self.stacked)
        layout_principal.addLayout(barra_navegacion)

    def _conectar_senales(self):
        # ── Navegación ──
        self.btn_operacion.clicked.connect(lambda: self._cambiar_vista(0))
        self.btn_manual.clicked.connect(lambda: self._cambiar_vista(1))
        self.btn_mantenimiento.clicked.connect(lambda: self._cambiar_vista(2))
        self.btn_referencias.clicked.connect(lambda: self._cambiar_vista(3))

        # ── Mantenimiento: calibrar referencia (paso 1) ──
        self.vista_mantenimiento.btn_calibrar_referencia.clicked.connect(
            self._calibrar_referencia
        )

        # ── Mantenimiento: recorte, cámara, inspección ──
        self.vista_mantenimiento.btn_definir_recorte.clicked.connect(
            self._definir_recorte
        )
        self.vista_mantenimiento.btn_capturar.clicked.connect(
            lambda: self._capturar_y_mostrar("mantenimiento")
        )
        self.vista_manual.sig_captura.connect(
            lambda: self._capturar_y_mostrar("manual")
        )
        self.vista_manual.sig_toggle_camara_viva.connect(self._toggle_camara_viva)

        self.vista_mantenimiento.btn_probar.clicked.connect(self._probar_inspeccion)
        self.vista_mantenimiento.btn_guardar_foto.clicked.connect(
            self._guardar_foto_prueba
        )

        # ── Modelo de color ──
        self.vista_mantenimiento.btn_agregar_entrenamiento.clicked.connect(
            self._agregar_imagen_entrenamiento
        )
        self.vista_mantenimiento.btn_entrenar.clicked.connect(self._entrenar_modelo)
        self.vista_mantenimiento.spin_umbral_pc.valueChanged.connect(
            self._actualizar_umbral_pc
        )

        # ── Operación: cambio de referencia desde el dropdown ──
        self.vista_operacion.sig_cambiar_referencia.connect(self._cambiar_referencia)

        # ── Operación: botón RUN → captura + inspección manual ──
        self.vista_operacion.sig_capturar_analizar.connect(self._capturar_y_analizar)

        # ── Vista Referencias: gestión ──
        self.vista_referencias.sig_seleccionar.connect(self._cambiar_referencia)
        self.vista_referencias.sig_crear.connect(self._crear_referencia)
        self.vista_referencias.sig_eliminar.connect(self._eliminar_referencia)

    # ── Ciclo de vida: carga inicial ───────────────────────────────────────────

    def _cargar_referencia_inicial(self):
        """
        Al arranque, activa la referencia guardada en config. Si no hay
        ninguna (o ya no existe en disco), muestra un aviso y deja todo
        en blanco — el usuario debe ir a 'Referencias' a resolverlo.
        """
        # Migración silenciosa del recorte: hasta v1.x el recorte vivía
        # dentro de meta.json de cada referencia.  Ahora es global en
        # calibracion.json.  Si calibracion.json todavía no tiene recorte
        # y la referencia activa sí lo trae, lo copiamos y borramos del
        # meta.  Una sola vez por instalación.
        self._migrar_recorte_global_si_aplica()

        nombre = self._config.referencia_activa

        if nombre and self._gestor.existe(nombre):
            self._aplicar_referencia(nombre)
            return

        if nombre and not self._gestor.existe(nombre):
            logging.warning(
                f"La referencia activa '{nombre}' ya no existe en disco."
            )
            self._config.referencia_activa = None
            self._config.guardar()

        # Sin referencia activa → refrescar UI vacía + aviso
        self._refrescar_listas_referencias()
        self._mostrar_aviso_sin_referencia()

    def _migrar_recorte_global_si_aplica(self):
        """
        Migración v1 → v2: el recorte pasó de `meta.json` por-referencia a
        `calibracion.json` global.  Si el global aún está vacío y alguna
        referencia trae recorte válido en su meta, copiamos la primera
        encontrada al global y limpiamos el campo `recorte` de TODOS los
        meta.json (para que no haya dos fuentes de verdad).
        """
        rec_global = self._config.recorte
        if rec_global["w"] > 0 and rec_global["h"] > 0:
            return  # ya está poblado, nada que migrar

        nombres = self._gestor.listar()
        rec_origen = None
        nombre_origen = None
        for n in nombres:
            meta = self._gestor.cargar_meta(n)
            r = meta.get("recorte") or {}
            if r.get("w", 0) > 0 and r.get("h", 0) > 0:
                rec_origen = r
                nombre_origen = n
                break

        if rec_origen is None:
            return  # ninguna referencia tiene recorte — instalación nueva

        self._config.recorte = rec_origen
        self._config.guardar()
        logging.info(
            "Migración: recorte copiado de meta.json('%s') a calibracion.json: %s",
            nombre_origen, rec_origen,
        )

        # Limpiar el campo de TODOS los meta.json para que no haya
        # información duplicada que pueda divergir.
        for n in nombres:
            meta = self._gestor.cargar_meta(n)
            if "recorte" in meta:
                meta.pop("recorte")
                self._gestor.guardar_meta(n, meta)

    def _mostrar_aviso_sin_referencia(self):
        hay_algunas = bool(self._gestor.listar())
        if hay_algunas:
            mensaje = (
                "No hay ninguna referencia activa.\n\n"
                "Ve a la pantalla 'Referencias' y selecciona una para\n"
                "comenzar la inspección."
            )
        else:
            mensaje = (
                "El sistema aún no tiene ninguna referencia de etiqueta.\n\n"
                "Ve a 'Referencias' y crea la primera. Después, en\n"
                "'Mantenimiento', define el área de inspección, agrega\n"
                "imágenes al banco y entrena el modelo."
            )
        QMessageBox.information(self, "Sin referencia activa", mensaje)

    # ── Cambio de referencia (handlers principales) ────────────────────────────

    def _cambiar_referencia(self, nombre: str):
        """Handler: el operador pidió activar otra referencia."""
        if not self._gestor.existe(nombre):
            QMessageBox.warning(
                self, "Referencia no existe",
                f"La referencia '{nombre}' ya no está en disco."
            )
            self._refrescar_listas_referencias()
            return

        if nombre == self._config.referencia_activa:
            return  # ya está activa, nada que hacer

        self._aplicar_referencia(nombre)
        self._config.guardar()  # persistir la selección para el próximo arranque

        self.vista_mantenimiento.lbl_estado.setText(
            f"Referencia activa cambiada a: {nombre}"
        )
        self.vista_operacion.reiniciar_contadores()

    def _aplicar_referencia(self, nombre: str):
        """
        Carga una referencia en memoria: aplica su recorte a la cámara,
        carga su .pkl en el inspector y refresca toda la UI dependiente.
        No persiste — el caller decide cuándo guardar.
        """
        meta = self._gestor.cargar_meta(nombre)

        # 1. Recorte de cámara — global, no por referencia.  Lo aplicamos
        # aquí también para asegurar que la cámara use el valor correcto
        # cuando se cambia de referencia (caso edge: si una vista anterior
        # lo había modificado por algún motivo).
        rec = self._config.recorte
        self._camara.establecer_recorte(rec["x"], rec["y"], rec["w"], rec["h"])

        # 2. Calibración por-referencia: rangos HSV (compatibilidad de firma —
        # la segmentación siempre usa NO_DOME) y ROIs semánticas de la paleta
        # auto-detectada.  Las ROIs sirven solo para NOMBRAR las zonas
        # defectuosas en los mensajes; si no hay calibración (referencia
        # recién creada), ambos quedan en None y el detector reporta las
        # celdas sin nombre de zona.
        carpeta = self._gestor.ruta_carpeta(nombre)
        cal_color = calibracion_referencia.cargar_calibracion_color(carpeta)
        self._rangos_hsv_activos = (
            calibracion_referencia.rangos_planos(cal_color) if cal_color else None
        )
        self._rois_activas = cal_color.get("rois") if cal_color else None

        # 3. Modelo + umbral, con las ROIs inyectadas al detector.
        ruta_pkl = self._gestor.ruta_modelo(nombre)
        self._inspector.preparar(
            ruta_modelo     = ruta_pkl if os.path.exists(ruta_pkl) else None,
            umbral_override = meta.get("umbral_modelo"),
            rangos_hsv      = self._rangos_hsv_activos,
            rois            = self._rois_activas,
        )

        # 4. Estado de config
        self._config.referencia_activa = nombre

        # 5. UI
        self._refrescar_listas_referencias()
        self._sincronizar_spinbox_umbral()
        self._actualizar_estado_banco()

        logging.info(
            f"Referencia activa: {nombre} (calibrada={cal_color is not None})"
        )

    def _refrescar_listas_referencias(self):
        """Refresca dropdown (Operación), lista (Referencias) y banner (Mant.)."""
        activa  = self._config.referencia_activa
        nombres = self._gestor.listar()

        self.vista_operacion.refrescar_referencias(nombres, activa)

        resumen = [(n, self._gestor.resumen_estado(n)) for n in nombres]
        self.vista_referencias.refrescar(resumen, activa)

        if activa:
            self.vista_mantenimiento.actualizar_referencia_activa(
                activa, self._gestor.resumen_estado(activa)
            )
        else:
            self.vista_mantenimiento.actualizar_referencia_activa(None)

    def _crear_referencia(self, nombre: str, descripcion: str):
        """Crea una nueva referencia (carpeta + meta.json vacío)."""
        exito, msg = self._gestor.crear(nombre, descripcion)
        if not exito:
            QMessageBox.warning(self, "No se pudo crear", msg)
            return

        # Primera referencia del sistema → activarla automáticamente.
        if not self._config.referencia_activa:
            self._aplicar_referencia(nombre)
            self._config.guardar()
            QMessageBox.information(
                self, "Referencia creada",
                f"Referencia '{nombre}' creada y activada.\n\n"
                "Próximos pasos:\n"
                "1. Ve a 'Mantenimiento' y define el área de inspección.\n"
                "2. Agrega 10–30 imágenes al banco de entrenamiento.\n"
                "3. Entrena el modelo de color."
            )
        else:
            self._refrescar_listas_referencias()
            QMessageBox.information(
                self, "Referencia creada",
                f"Referencia '{nombre}' creada.\n\n"
                "Selecciónala cuando quieras activarla. La referencia\n"
                f"activa sigue siendo: {self._config.referencia_activa}"
            )

    def _eliminar_referencia(self, nombre: str):
        """Borra la carpeta completa de una referencia."""
        era_activa = (nombre == self._config.referencia_activa)
        exito, msg = self._gestor.eliminar(nombre)

        if not exito:
            QMessageBox.critical(self, "Error", msg)
            return

        if era_activa:
            # Quedamos sin referencia activa — resetear inspector.  El recorte
            # de cámara NO se toca: es global y compartido entre referencias.
            self._config.referencia_activa = None
            self._config.guardar()
            self._inspector.preparar(ruta_modelo=None)

        self._refrescar_listas_referencias()

        if era_activa:
            self.vista_mantenimiento.lbl_estado.setText(
                f"Referencia '{nombre}' (activa) eliminada — sin referencia activa."
            )
            QMessageBox.information(
                self, "Referencia eliminada",
                f"La referencia '{nombre}' estaba activa y fue eliminada.\n\n"
                "Selecciona o crea otra referencia para continuar."
            )
        else:
            self.vista_mantenimiento.lbl_estado.setText(
                f"Referencia '{nombre}' eliminada."
            )

    # ── Navegación ─────────────────────────────────────────────────────────────

    _ESTILO_NAV_BASE = """
        QPushButton {
            font-size: 16px; font-weight: bold;
            border: 2px solid #cccccc; border-radius: 8px;
            padding: 8px 16px; background-color: #f0f0f0;
            color: #222222;
        }
        QPushButton:pressed { background-color: #d0d0d0; }
    """
    _ESTILO_NAV_ACTIVO = """
        QPushButton {
            font-size: 16px; font-weight: bold;
            border: 2px solid #2E7D32; border-radius: 8px;
            padding: 8px 16px; background-color: #4CAF50;
            color: white;
        }
    """

    def _actualizar_botones_navegacion(self, indice_activo: int):
        """Resalta en verde el botón de la vista activa, neutro los demás."""
        for i, btn in enumerate(self._botones_nav):
            if i == indice_activo:
                btn.setStyleSheet(self._ESTILO_NAV_ACTIVO)
            else:
                btn.setStyleSheet(self._ESTILO_NAV_BASE)

    def _cambiar_vista(self, indice):
        if indice != 1:
            self._detener_camara_viva()
        self.stacked.setCurrentIndex(indice)
        self._actualizar_botones_navegacion(indice)
        if indice == 0:
            self.vista_operacion.reiniciar_contadores()
        elif indice == 3:
            # Al entrar en Referencias, refrescamos por si otra parte del
            # programa cambió algo en disco.
            self._refrescar_listas_referencias()

    # ── Cámara ─────────────────────────────────────────────────────────────────

    def _capturar_y_mostrar(self, destino: str):
        try:
            frame_bgr = self._camara.capturar()
        except RuntimeError as e:
            QMessageBox.critical(self, "Error de cámara", str(e))
            return

        self._ultimo_frame = frame_bgr

        pixmap = self._frame_a_pixmap(frame_bgr)
        lbl = (self.vista_mantenimiento.lbl_imagen
               if destino == "mantenimiento"
               else self.vista_manual.lbl_preview)
        lbl.setPixmap(
            pixmap.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ── Cámara en vivo (vista Manual) ─────────────────────────────────────────

    def _toggle_camara_viva(self):
        if self._timer_camara_viva.isActive():
            self._detener_camara_viva()
        else:
            self._timer_camara_viva.start()

    def _tick_camara_viva(self):
        """Captura un frame y lo muestra en el preview de Manual."""
        try:
            frame_bgr = self._camara.capturar()
        except RuntimeError:
            self._detener_camara_viva()
            return
        lbl = self.vista_manual.lbl_preview
        pixmap = self._frame_a_pixmap(frame_bgr)
        lbl.setPixmap(
            pixmap.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _detener_camara_viva(self):
        """Para el timer y sincroniza el estado del botón."""
        if self._timer_camara_viva.isActive():
            self._timer_camara_viva.stop()
        self.vista_manual.resetear_camara_viva()

    # ── Captura + inspección manual (botón RUN de Operación) ─────────────────

    def _capturar_y_analizar(self):
        """
        Flujo manual del prototipo: captura una foto, la inspecciona con el
        modelo de la referencia activa y actualiza la vista de Operación.
        """
        if not self._config.referencia_activa:
            QMessageBox.warning(
                self, "Sin referencia activa",
                "Selecciona una referencia antes de capturar."
            )
            return

        try:
            frame_bgr = self._camara.capturar()
        except RuntimeError as e:
            QMessageBox.critical(self, "Error de cámara", str(e))
            return

        self._ultimo_frame = frame_bgr

        resultado = self._inspector.inspeccionar(frame_bgr)

        # Mostrar preferentemente la imagen rectificada — así el operador ve
        # exactamente el área canónica que evaluó el modelo.  Si la
        # rectificación falló (ej. sin_luz o sin segmentar), cae al frame
        # crudo para que de todas formas haya algo que ver.
        imagen_mostrar = resultado.get("imagen_rectificada")
        if imagen_mostrar is None:
            imagen_mostrar = frame_bgr
        self.vista_operacion.actualizar_imagen(self._frame_a_pixmap(imagen_mostrar))

        self.vista_operacion.mostrar_resultado(resultado)

    # ── Área de inspección (recorte global) ─────────────────────────────────────

    def _definir_recorte(self):
        """
        Define el recorte global de cámara — la zona del frame donde caben
        las etiquetas cuando entran al domo.  Es global, no por referencia,
        porque todas las etiquetas usan el mismo dispensador y el mismo domo.
        Se calibra una sola vez al instalar el equipo y luego raramente.
        """
        try:
            frame_completo = self._camara.capturar(aplicar_recorte=False)
        except RuntimeError as e:
            QMessageBox.critical(self, "Error de cámara", str(e))
            return

        QMessageBox.information(
            self, "Instrucciones — Área de inspección",
            "Se abrirá la imagen completa de la cámara.  Define el recuadro\n"
            "donde caben las etiquetas dentro del domo — TODAS las\n"
            "referencias compartirán este recorte.\n\n"
            "• Arrastra el mouse para seleccionar el área\n"
            "• ENTER o ESPACIO → confirmar selección\n"
            "• ESC → cancelar sin guardar cambios\n\n"
            "Coloca una etiqueta bajo la cámara antes de continuar."
        )

        MAX_VIS_W = 900
        MAX_VIS_H = 700
        alto_orig, ancho_orig = frame_completo.shape[:2]
        factor_escala = min(MAX_VIS_W / ancho_orig, MAX_VIS_H / alto_orig, 1.0)
        ancho_vis = int(ancho_orig * factor_escala)
        alto_vis  = int(alto_orig  * factor_escala)
        imagen_vis = cv2.resize(frame_completo, (ancho_vis, alto_vis),
                                interpolation=cv2.INTER_LINEAR)

        rec_actual = self._config.recorte
        if rec_actual["w"] > 0 and rec_actual["h"] > 0:
            rx = int(rec_actual["x"] * factor_escala)
            ry = int(rec_actual["y"] * factor_escala)
            rw = int(rec_actual["w"] * factor_escala)
            rh = int(rec_actual["h"] * factor_escala)
            cv2.rectangle(imagen_vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 100), 2)
            cv2.putText(imagen_vis, "area actual", (rx + 4, ry + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)

        nombre_ventana = "Definir area de inspeccion - ESC para cancelar"
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, ancho_vis, alto_vis)
        x_vis, y_vis, w_vis, h_vis = cv2.selectROI(
            nombre_ventana, imagen_vis, fromCenter=False, showCrosshair=True
        )
        cv2.destroyAllWindows()

        if w_vis == 0 or h_vis == 0:
            self.vista_mantenimiento.lbl_estado.setText(
                "Estado: definición de área cancelada."
            )
            return

        factor_inverso = 1.0 / factor_escala
        x_orig = max(0, min(int(x_vis * factor_inverso), ancho_orig - 1))
        y_orig = max(0, min(int(y_vis * factor_inverso), alto_orig  - 1))
        w_orig = min(int(w_vis * factor_inverso), ancho_orig - x_orig)
        h_orig = min(int(h_vis * factor_inverso), alto_orig  - y_orig)

        # Advertir si hay modelos entrenados de cualquier referencia: cambiar
        # recorte invalida features aprendidas en TODAS porque el encuadre
        # cambia para cualquier etiqueta capturada en adelante.
        nombres_entrenadas = [
            n for n in self._gestor.listar() if self._gestor.tiene_modelo(n)
        ]
        if nombres_entrenadas:
            respuesta = QMessageBox.warning(
                self, "Recalibración necesaria",
                f"Hay {len(nombres_entrenadas)} referencia(s) con modelo de color "
                f"entrenado: {', '.join(nombres_entrenadas)}.\n\n"
                "Cambiar el área de inspección invalida TODOS los modelos\n"
                "porque el encuadre de las etiquetas será distinto.\n\n"
                "Deberás recapturar el banco y reentrenar cada referencia.\n\n"
                "¿Continuar de todas formas?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if respuesta == QMessageBox.Cancel:
                return

        # Guardar en config global.
        self._config.recorte = {"x": x_orig, "y": y_orig, "w": w_orig, "h": h_orig}
        self._config.guardar()
        self._camara.establecer_recorte(x_orig, y_orig, w_orig, h_orig)
        self._capturar_y_mostrar("mantenimiento")

        self.vista_mantenimiento.lbl_estado.setText(
            f"Área global: {w_orig}×{h_orig} px  (origen x={x_orig}, y={y_orig})"
        )
        QMessageBox.information(
            self, "Área de inspección definida",
            f"Recorte global: {w_orig}×{h_orig} px desde ({x_orig}, {y_orig}).\n\n"
            "Todas las referencias usan este recorte — basta calibrarlo una vez.\n\n"
            "Próximos pasos para la referencia activa:\n"
            "1. 'Calibrar referencia' — el sistema captura una maestra y "
            "detecta automáticamente la paleta de colores\n"
            "2. Capturar imágenes al banco\n"
            "3. Entrenar el modelo de color"
        )

    # ── Probar inspección ──────────────────────────────────────────────────────

    def _probar_inspeccion(self):
        if not self._config.referencia_activa:
            QMessageBox.warning(self, "Sin referencia activa",
                                "Selecciona una referencia antes de probar.")
            return

        self._capturar_y_mostrar("mantenimiento")
        if self._ultimo_frame is None:
            return

        resultado = self._inspector.inspeccionar_diagnostico(self._ultimo_frame)

        imagen_anotada = self._anotar_imagen_prueba(self._ultimo_frame, resultado)
        lbl_img = self.vista_mantenimiento.lbl_imagen
        lbl_img.setPixmap(
            self._frame_a_pixmap(imagen_anotada).scaled(
                lbl_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        lbl = self.vista_mantenimiento.lbl_resultado_prueba
        defecto = resultado.get("defecto")
        if not resultado.get("disponible", False):
            lbl.setText("MODELO NO ENTRENADO — entrena el modelo de color antes de probar")
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #757575; border-radius: 6px; padding: 6px;"
            )
        elif defecto == "sin_luz":
            lbl.setText("RECHAZADA ✗  — imagen oscura (luz apagada o sin etiqueta)")
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #E65100; border-radius: 6px; padding: 6px;"
            )
        elif defecto == "sin_etiqueta":
            # El motivo específico se deriva del detalle que viene del
            # detector — mismo mapeo que el popup de "mostrar_detalle_prueba"
            # para que la franja y el diálogo sean consistentes.
            detalle = (resultado.get("detalle") or "").lower()
            if "paleta" in detalle:
                texto = "RECHAZADA ✗  — paleta de colores distinta"
            elif "fuera del área" in detalle or "cortada" in detalle:
                texto = "RECHAZADA ✗  — etiqueta cortada por el borde del domo"
            elif "brillo" in detalle:
                texto = "RECHAZADA ✗  — brillo fuera del rango aprendido"
            elif ("segmentar" in detalle or "contorno" in detalle
                    or "pequeña" in detalle):
                texto = "RECHAZADA ✗  — sin etiqueta detectada en el domo"
            else:
                texto = "RECHAZADA ✗  — firma estructural fuera de rango"
            lbl.setText(texto)
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #E65100; border-radius: 6px; padding: 6px;"
            )
        elif defecto == "color":
            lbl.setText("RECHAZADA ✗  — mancha o color fuera de rango")
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #f44336; border-radius: 6px; padding: 6px;"
            )
        elif resultado["ok"]:
            lbl.setText("RESULTADO: OK ✓  — etiqueta dentro de lo aprendido")
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #4CAF50; border-radius: 6px; padding: 6px;"
            )
        else:
            lbl.setText("RESULTADO: DEFECTO ✗  — ver detalle")
            lbl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: white; "
                "background-color: #f44336; border-radius: 6px; padding: 6px;"
            )

        self.vista_mantenimiento.mostrar_detalle_prueba(resultado)
        self.vista_mantenimiento.lbl_estado.setText(
            "Estado: prueba de diagnóstico completada"
        )

    # ── Guardar foto de prueba ─────────────────────────────────────────────────

    def _guardar_foto_prueba(self):
        if self._ultimo_frame is None:
            QMessageBox.warning(self, "Sin imagen", "Captura una imagen primero.")
            return

        nombre, ok = QInputDialog.getText(
            self, "Guardar foto de prueba", "Nombre del archivo (sin extensión):"
        )
        if not ok or not nombre.strip():
            return

        nombre = nombre.strip()
        for c in r'\/:*?"<>|':
            nombre = nombre.replace(c, "_")

        carpeta = os.path.join(_RUTA_BASE, "fotos_prueba")
        os.makedirs(carpeta, exist_ok=True)

        ruta = os.path.join(carpeta, f"{nombre}.jpg")
        contador = 1
        while os.path.exists(ruta):
            ruta = os.path.join(carpeta, f"{nombre}_{contador}.jpg")
            contador += 1

        imwrite_safe(ruta, self._ultimo_frame)
        self.vista_mantenimiento.lbl_estado.setText(
            f"Foto guardada: fotos_prueba/{os.path.basename(ruta)}"
        )

    # ── Calibración de referencia (paso 1, antes del banco) ───────────────────

    def _calibrar_referencia(self):
        """
        Lanza el flujo de calibración automática de la referencia activa.

        Pipeline:
            1. Captura una maestra del domo.
            2. Rectifica al canvas canónico usando el rango "no-dome"
               universal (V≥50) — funciona para cualquier paleta porque
               sólo distingue label vs fondo negro del domo.
            3. Abre `DialogoCalibracion`, que detecta automáticamente la
               paleta con k-means HSV y deja que el operador revise y
               edite las ROIs antes de guardar.

        El operador NO pinta ROIs — el sistema descubre la paleta solo.

        Tras guardar la calibración, recargamos la referencia para que el
        pipeline empiece a usar los rangos y las ROIs recién definidos.
        """
        activa = self._config.referencia_activa
        if not activa:
            QMessageBox.warning(
                self, "Sin referencia activa",
                "Selecciona o crea una referencia antes de calibrar."
            )
            return

        try:
            frame = self._camara.capturar()
        except RuntimeError as e:
            QMessageBox.critical(self, "Error de cámara", str(e))
            return

        # Rectificación con el rango UNIVERSAL "no-dome".  Independiente de
        # la paleta — sólo necesita que el dome sea oscuro (V<40 ∧ S<20),
        # que es siempre el caso en este prototipo.  Así no dependemos de
        # tener calibración previa para poder calibrar.
        rectificada, razon = rectificar_etiqueta(
            frame, rangos_hsv=RANGOS_HSV_NO_DOME
        )
        if rectificada is None:
            detalles = {
                "sin_segmentacion":  "No se pudo separar la etiqueta del fondo "
                                     "del domo.  Verifica iluminación y centrado.",
                "contorno_vacio":    "La máscara no produjo contornos.",
                "contorno_diminuto": "La etiqueta se ve demasiado pequeña.",
                "etiqueta_cortada":  "La etiqueta toca el borde del área de "
                                     "inspección — céntrala dentro del domo "
                                     "o ajusta el recorte global.",
            }
            QMessageBox.warning(
                self, "No se pudo preparar la maestra",
                detalles.get(razon, f"Rectificación falló ({razon}).")
            )
            return

        carpeta = self._gestor.ruta_carpeta(activa)
        dialogo = DialogoCalibracion(self, activa, carpeta, rectificada)
        if dialogo.exec_() == dialogo.Accepted:
            # Recargar la referencia para activar los rangos y ROIs nuevos.
            # Si el operador cancela no hay nada que restaurar: el inspector
            # conserva su modelo cargado (esta variante no necesita liberar
            # memoria antes de abrir el diálogo).
            self._aplicar_referencia(activa)
            self.vista_mantenimiento.lbl_estado.setText(
                f"Referencia '{activa}' calibrada — ahora puedes capturar el banco."
            )

    # ── Modelo de color: banco de entrenamiento ───────────────────────────────

    def _agregar_imagen_entrenamiento(self):
        """
        Guarda la imagen RECTIFICADA en referencias/<activa>/buenas/.

        Se guarda la vista rectificada (canvas canónico) en lugar del
        frame crudo por dos razones:

        1. Detección temprana de problemas: si la rectificación sale mal
           (rotación incorrecta, segmentación fallida, etiqueta cortada),
           el operador lo ve en la miniatura del archivo ANTES de
           entrenar y puede descartar la muestra.  En el esquema anterior
           la rotación mala solo se descubría al inspeccionar el .pkl.
        2. Ahorra un warp en el entrenamiento: `rectificar_si_necesario`
           detecta que la imagen ya está en el canvas canónico y la
           devuelve tal cual.

        Si rectificar falla, la imagen NO se guarda y el operador ve un
        mensaje con el motivo específico (cortada, sin segmentar, etc.)
        para poder ajustar iluminación/posición y volver a capturar.
        """
        activa = self._config.referencia_activa
        if not activa:
            QMessageBox.warning(self, "Sin referencia activa",
                                "Selecciona una referencia antes de agregar imágenes.")
            return

        if self._ultimo_frame is None:
            QMessageBox.warning(self, "Sin imagen",
                                "Primero captura una imagen con 'Capturar imagen'.")
            return

        # Rectificar ANTES de guardar.  Si falla, informar el motivo concreto.
        rectificada, razon = rectificar_etiqueta(
            self._ultimo_frame, rangos_hsv=self._rangos_hsv_activos
        )
        if rectificada is None:
            detalles = {
                "sin_segmentacion":  "No se pudo aislar la etiqueta del fondo — "
                                     "revisa iluminación del domo.",
                "contorno_vacio":    "La máscara no produjo contornos.",
                "contorno_diminuto": "La etiqueta se ve demasiado pequeña "
                                     "en el frame.",
                "etiqueta_cortada":  "La etiqueta apoya el borde del área de "
                                     "inspección — céntrala dentro del domo.",
            }
            QMessageBox.warning(
                self, "No se pudo rectificar la imagen",
                detalles.get(razon, f"Rectificación falló ({razon})."
                             " Captura otra y vuelve a intentar.")
            )
            return

        banco = self._gestor.ruta_banco(activa)
        os.makedirs(banco, exist_ok=True)

        # Siguiente índice = max(índices existentes) + 1, no count+1.
        # Si se borra una imagen intermedia (ej. buena_006.jpg), el count
        # deja de coincidir con el último número usado y el archivo nuevo
        # sobrescribe uno existente.  Este patrón funciona con gaps.
        indices = []
        for nombre_archivo in os.listdir(banco):
            if not nombre_archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = re.match(r"^buena_(\d+)\.", nombre_archivo, re.IGNORECASE)
            if m:
                indices.append(int(m.group(1)))
        siguiente = (max(indices) + 1) if indices else 1
        ruta = os.path.join(banco, f"buena_{siguiente:03d}.jpg")

        if imwrite_safe(ruta, rectificada):
            self._actualizar_estado_banco()
            self._refrescar_listas_referencias()
            self.vista_mantenimiento.lbl_estado.setText(
                f"Imagen agregada al banco: {os.path.basename(ruta)} (rectificada)"
            )
        else:
            QMessageBox.critical(self, "Error",
                                 f"No se pudo guardar la imagen en:\n{ruta}")

    def _actualizar_estado_banco(self):
        """Actualiza el contador de imágenes en el grupo del modelo de color."""
        activa = self._config.referencia_activa
        if not activa:
            self.vista_mantenimiento.lbl_banco_estado.setText(
                "Banco: — (sin referencia activa)"
            )
            return

        n        = self._gestor.contar_imagenes_banco(activa)
        detector = self._inspector.detector

        if detector.entrenado:
            estado = (f"Banco: {n} imágenes  |  "
                      f"Modelo: cargado (umbral={detector.umbral:.4f})")
        elif self._gestor.tiene_modelo(activa):
            estado = (f"Banco: {n} imágenes  |  "
                      "Modelo: en disco (recarga la referencia para usarlo)")
        else:
            estado = f"Banco: {n} imagen(es)  |  Modelo: no entrenado"

        self.vista_mantenimiento.lbl_banco_estado.setText(estado)

    def _sincronizar_spinbox_umbral(self):
        """Refleja el umbral actual del detector en el spinbox (sin disparar signal)."""
        detector = self._inspector.detector
        if detector.entrenado and detector.umbral is not None:
            self.vista_mantenimiento.spin_umbral_pc.blockSignals(True)
            self.vista_mantenimiento.spin_umbral_pc.setValue(detector.umbral)
            self.vista_mantenimiento.spin_umbral_pc.blockSignals(False)

    # ── Modelo de color: entrenamiento ─────────────────────────────────────────

    def _entrenar_modelo(self):
        """Lanza el entrenamiento del modelo de color en un hilo secundario."""
        activa = self._config.referencia_activa
        if not activa:
            QMessageBox.warning(self, "Sin referencia activa",
                                "Selecciona una referencia antes de entrenar.")
            return

        # Pre-condición: la referencia debe estar calibrada (paleta de
        # color).  Las ROIs de la calibración le dan nombre a las zonas
        # defectuosas ("mancha en banner verde") en los mensajes de
        # inspección — sin ellas el operador solo vería coordenadas.
        carpeta = self._gestor.ruta_carpeta(activa)
        if not calibracion_referencia.referencia_calibrada(carpeta):
            QMessageBox.warning(
                self, "Referencia sin calibrar",
                "Esta referencia aún no fue calibrada.\n\n"
                "Pulsa 'Calibrar referencia' (paso 1) para detectar la\n"
                "paleta de colores antes de entrenar.\n\n"
                "Sin calibración el sistema no sabría cómo nombrar las\n"
                "zonas de la etiqueta al reportar defectos.",
            )
            return

        if (self._hilo_entrenamiento is not None
                and self._hilo_entrenamiento.isRunning()):
            return

        banco = self._gestor.ruta_banco(activa)
        if not os.path.isdir(banco):
            QMessageBox.warning(
                self, "Banco vacío",
                "No hay imágenes en el banco de entrenamiento.\n\n"
                "Captura etiquetas buenas y usa 'Agregar imagen al banco'."
            )
            return

        rutas = [
            os.path.join(banco, f)
            for f in os.listdir(banco)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not rutas:
            QMessageBox.warning(
                self, "Banco vacío",
                "No hay imágenes en el banco de entrenamiento.\n\n"
                "Captura etiquetas buenas y usa 'Agregar imagen al banco'."
            )
            return

        if len(rutas) < 5:
            respuesta = QMessageBox.question(
                self, "Pocas imágenes",
                f"Solo hay {len(rutas)} imagen(es) en el banco.\n"
                "Se recomiendan al menos 10–30 para resultados confiables.\n\n"
                "¿Continuar de todas formas?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
            )
            if respuesta == QMessageBox.Cancel:
                return

        self.vista_mantenimiento.btn_entrenar.setEnabled(False)

        # El callback del detector emite (porcentaje 0-100, mensaje).  El
        # entrenamiento por celdas tarda <1 s, así que el diálogo es casi
        # simbólico — pero conserva el patrón modal por si el banco crece
        # o el disco responde lento.
        self._dialogo_progreso = QProgressDialog(
            "Analizando el banco de imágenes...", None, 0, 100, self
        )
        self._dialogo_progreso.setWindowTitle("Entrenando modelo de color")
        self._dialogo_progreso.setWindowModality(Qt.WindowModal)
        self._dialogo_progreso.setCancelButton(None)
        self._dialogo_progreso.setAutoClose(False)
        self._dialogo_progreso.setAutoReset(False)
        self._dialogo_progreso.show()

        def _actualizar_progreso(pct, msg):
            if self._dialogo_progreso is None:
                return
            self._dialogo_progreso.setValue(pct)
            if msg:
                self._dialogo_progreso.setLabelText(msg)

        # Detector nuevo para cada entrenamiento — así el banco nuevo
        # reemplaza por completo al anterior (no heredamos estado).  Las
        # ROIs semánticas de la calibración solo sirven para nombrar las
        # zonas defectuosas en los mensajes; la detección es por grilla.
        # No hace falta liberar memoria antes: sin GPU ni modelos pesados,
        # el footprint del entrenamiento es de unos pocos MB.
        detector_nuevo = DetectorROI()
        detector_nuevo.set_calibracion(self._rois_activas)
        ruta_salida    = self._gestor.ruta_modelo(activa)

        self._hilo_entrenamiento = _HiloEntrenamiento(
            detector_nuevo, rutas, ruta_salida
        )
        self._hilo_entrenamiento.progreso.connect(_actualizar_progreso)
        self._hilo_entrenamiento.terminado.connect(
            lambda exito, msg: self._entrenamiento_terminado(exito, msg, activa)
        )
        self._hilo_entrenamiento.start()

        self.vista_mantenimiento.lbl_estado.setText(
            f"Entrenando '{activa}' con {len(rutas)} imágenes... espera."
        )

    def _entrenamiento_terminado(self, exito: bool, mensaje: str,
                                  ref_entrenada: str):
        if self._dialogo_progreso is not None:
            self._dialogo_progreso.close()
            self._dialogo_progreso = None

        self.vista_mantenimiento.btn_entrenar.setEnabled(True)

        if not exito:
            QMessageBox.critical(self, "Error en entrenamiento", mensaje)
            self.vista_mantenimiento.lbl_estado.setText("Estado: entrenamiento fallido")
            return

        # Si la referencia activa sigue siendo la que entrenamos, recargar el
        # .pkl desde disco en el inspector y limpiar cualquier override viejo.
        if ref_entrenada == self._config.referencia_activa:
            meta = self._gestor.cargar_meta(ref_entrenada)
            meta["umbral_modelo"] = None  # el nuevo modelo trae su umbral auto
            self._gestor.guardar_meta(ref_entrenada, meta)

            self._inspector.preparar(
                ruta_modelo     = self._gestor.ruta_modelo(ref_entrenada),
                umbral_override = None,
                rangos_hsv      = self._rangos_hsv_activos,
                rois            = self._rois_activas,
            )
            self._sincronizar_spinbox_umbral()

        self._actualizar_estado_banco()
        self._refrescar_listas_referencias()

        detector = self._inspector.detector
        umbral_txt = (f"{detector.umbral:.4f}"
                      if detector.entrenado and detector.umbral is not None
                      else "—")
        self.vista_mantenimiento.lbl_estado.setText(
            f"Entrenamiento de '{ref_entrenada}' completado — umbral={umbral_txt}"
        )
        QMessageBox.information(self, "Entrenamiento completado", mensaje)

    def _actualizar_umbral_pc(self, valor: float):
        """Aplica el umbral del spinbox al detector y lo guarda en meta.json."""
        activa = self._config.referencia_activa
        if not activa:
            self.vista_mantenimiento.lbl_estado.setText(
                "No hay referencia activa — selecciona una antes de cambiar el umbral."
            )
            return

        if not self._inspector.detector.entrenado:
            self.vista_mantenimiento.lbl_estado.setText(
                f"Umbral guardado ({valor:.4f}) — entrena el modelo para aplicarlo."
            )
        else:
            self.vista_mantenimiento.lbl_estado.setText(
                f"Umbral actualizado: {valor:.4f}"
            )

        self._inspector.detector.umbral = float(valor)
        meta = self._gestor.cargar_meta(activa)
        meta["umbral_modelo"] = float(valor)
        self._gestor.guardar_meta(activa, meta)
        # Refresca la línea "Banco: N imágenes | Modelo: cargado (umbral=…)"
        # para que el valor mostrado a la derecha del spinbox refleje el
        # umbral recién aplicado.
        self._actualizar_estado_banco()
        logging.info(
            f"Umbral del modelo de color actualizado a {valor:.4f} para '{activa}'."
        )

    # ── Anotación de imagen de prueba ──────────────────────────────────────────

    def _anotar_imagen_prueba(self, frame_bgr, resultado: dict):
        """
        Devuelve la imagen a mostrar tras inspeccionar. Si el pipeline llegó
        a rectificar la etiqueta, muestra esa vista canónica — así el
        técnico ve exactamente el área que se evaluó. Si no hubo
        rectificación (rechazo temprano por brillo o segmentación fallida),
        cae al frame original.

        Si el detector reportó celdas fuera de rango (defecto="color"), se
        dibujan como rectángulos rojos sobre la rectificada — el técnico ve
        DÓNDE está la mancha/desvío, no solo que existe.
        """
        rectificada = resultado.get("imagen_rectificada")
        if rectificada is None:
            return frame_bgr
        celdas = resultado.get("celdas_malas")
        if celdas:
            rectificada = rectificada.copy()
            for celda in celdas:
                x, y, w, h = celda["bbox"]
                cv2.rectangle(rectificada, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
        return rectificada

    # ── Eventos de teclado y cierre ────────────────────────────────────────────

    def keyPressEvent(self, evento: QKeyEvent):
        if evento.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(evento)

    def closeEvent(self, evento):
        logging.info("Cerrando aplicación — liberando recursos.")

        self._timer_camara_viva.stop()

        if (self._hilo_entrenamiento is not None
                and self._hilo_entrenamiento.isRunning()):
            logging.info("Esperando a que termine el entrenamiento en curso...")
            self._hilo_entrenamiento.wait(3000)  # hasta 3s

        try:
            self._config.guardar()
        except Exception as e:
            logging.warning(f"No se pudo guardar la configuración al cerrar: {e}")

        self._camara.cerrar()
        evento.accept()

    # ── Auxiliar ───────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_a_pixmap(frame_bgr) -> QPixmap:
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img_qt = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(img_qt)
