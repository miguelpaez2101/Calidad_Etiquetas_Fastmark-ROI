"""
VistaOperacion - La pantalla principal del operador (Prototipo)
=================================================================
Esta es la pantalla que el operador ve todo el día.
Muestra:
    - Dropdown arriba para cambiar de referencia de etiqueta
    - La última imagen capturada (grande, al centro)
    - El resultado: OK (verde) o DEFECTO (rojo)
    - Tres botones: Home, Run, Stop — en el prototipo sólo RUN está activo
      y actúa como "Capturar y analizar" (los otros dos quedan deshabilitados
      hasta que el prototipo avance a banda automática)
    - Conteo de etiquetas: buenas / malas / total
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal


def _texto_motivo_rechazo(defecto: str, detalle: str) -> str:
    """
    Mapea (defecto, detalle) a un mensaje corto para el indicador grande
    de Operación.  Usa las mismas keywords que
    vista_mantenimiento.mostrar_detalle_prueba para mantener consistencia
    entre ambas pantallas.
    """
    if defecto == "sin_luz":
        return "SIN LUZ"
    d = (detalle or "").lower()
    if "paleta" in d:
        return "PALETA DISTINTA"
    if "fuera del área" in d or "cortada" in d:
        return "ETIQUETA CORTADA"
    if "brillo" in d:
        return "BRILLO FUERA DE RANGO"
    if "segmentar" in d or "contorno" in d or "pequeña" in d:
        return "SIN ETIQUETA"
    return "FIRMA NO COINCIDE"


class VistaOperacion(QWidget):
    """
    La vista de operación. Hereda de QWidget (un panel genérico).

    Señales:
        sig_capturar_analizar   — tocaron "Run" (captura + inspección manual)
        sig_cambiar_referencia  — el operador escogió otra referencia del dropdown
    """

    sig_capturar_analizar = pyqtSignal()
    sig_cambiar_referencia = pyqtSignal(str)  # nombre de la referencia

    def __init__(self):
        super().__init__()

        # Contadores internos
        self.total_buenas = 0
        self.total_malas  = 0

        # Bandera para ignorar la señal del QComboBox cuando lo rellenamos
        # programáticamente (sino entraríamos en loop infinito).
        self._bloquear_signal_combo = False

        self._crear_interfaz()

    def _crear_interfaz(self):
        """Construye toda la interfaz de la vista de operación."""

        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Imagen (columna izquierda, domina el ancho) ───────────────────────
        self.lbl_imagen = QLabel("Sin imagen")
        self.lbl_imagen.setAlignment(Qt.AlignCenter)
        self.lbl_imagen.setMinimumHeight(420)
        self.lbl_imagen.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                color: #666666;
                font-size: 18px;
                border: 2px solid #333333;
                border-radius: 8px;
            }
        """)

        # ── Panel derecho (referencia + resultado + conteo + botones) ────────
        panel = QWidget()
        panel.setFixedWidth(360)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        # Referencia activa
        lbl_ref = QLabel("Referencia:")
        lbl_ref.setStyleSheet("font-size: 14px; font-weight: bold; padding: 2px;")

        self.combo_referencia = QComboBox()
        self.combo_referencia.setMinimumHeight(40)
        self.combo_referencia.setStyleSheet("""
            QComboBox {
                font-size: 15px; padding: 4px 10px;
                border: 2px solid #333333; border-radius: 6px;
                background-color: #1a1a2e; color: #e0e0e0;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1a1a2e; color: #e0e0e0;
                selection-background-color: #2e4e6e;
            }
        """)
        self.combo_referencia.currentIndexChanged.connect(self._al_cambiar_combo)

        # Indicador de resultado
        self.lbl_resultado = QLabel("ESPERANDO")
        self.lbl_resultado.setAlignment(Qt.AlignCenter)
        self.lbl_resultado.setMinimumHeight(80)
        self.lbl_resultado.setWordWrap(True)
        self.lbl_resultado.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #888888;
                background-color: #333333;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # Conteo
        self.lbl_conteo = QLabel("Buenas: 0  |  Malas: 0  |  Total: 0")
        self.lbl_conteo.setAlignment(Qt.AlignCenter)
        self.lbl_conteo.setStyleSheet("font-size: 15px; padding: 5px;")

        # Botones HOME / RUN / STOP — apilados vertical
        self.btn_home = QPushButton("HOME")
        self.btn_run  = QPushButton("RUN")
        self.btn_stop = QPushButton("STOP")

        self.btn_home.setMinimumHeight(64)
        self.btn_home.setStyleSheet("""
            QPushButton {
                font-size: 20px; font-weight: bold;
                background-color: #2196F3; color: white;
                border-radius: 10px; padding: 10px 30px;
            }
            QPushButton:pressed { background-color: #1976D2; }
            QPushButton:disabled { background-color: #546E7A; color: #AAAAAA; }
        """)

        self.btn_run.setMinimumHeight(80)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-size: 24px; font-weight: bold;
                background-color: #4CAF50; color: white;
                border-radius: 10px; padding: 10px 30px;
            }
            QPushButton:pressed { background-color: #388E3C; }
        """)

        self.btn_stop.setMinimumHeight(64)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                font-size: 20px; font-weight: bold;
                background-color: #f44336; color: white;
                border-radius: 10px; padding: 10px 30px;
            }
            QPushButton:pressed { background-color: #D32F2F; }
            QPushButton:disabled { background-color: #6D4C41; color: #AAAAAA; }
        """)

        # En el prototipo sólo RUN está activo (captura y analiza manualmente).
        # Home y Stop quedan deshabilitados hasta que haya banda transportadora.
        self.btn_home.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # Armado del panel — referencia arriba, resultado y conteo en medio,
        # botones abajo con el RUN destacado en el centro de la columna.
        col.addWidget(lbl_ref)
        col.addWidget(self.combo_referencia)
        col.addWidget(self.lbl_resultado)
        col.addWidget(self.lbl_conteo)
        col.addStretch(1)
        col.addWidget(self.btn_home)
        col.addWidget(self.btn_run)
        col.addWidget(self.btn_stop)

        # ── Contenido principal: imagen izq + panel derecho ───────────────────
        contenido = QHBoxLayout()
        contenido.setSpacing(12)
        contenido.addWidget(self.lbl_imagen, stretch=1)
        contenido.addWidget(panel)

        layout.addLayout(contenido)

        # ── Conectar botones a señales ────────────────────────────────────────
        # En el prototipo RUN dispara una captura + inspección manual.
        self.btn_run.clicked.connect(self.sig_capturar_analizar.emit)

    # ── API pública para MainWindow ────────────────────────────────────────────

    def actualizar_imagen(self, pixmap):
        """Recibe un QPixmap y lo muestra escalado."""
        imagen_escalada = pixmap.scaled(
            self.lbl_imagen.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_imagen.setPixmap(imagen_escalada)

    def mostrar_resultado(self, resultado):
        """Actualiza el indicador OK/DEFECTO y los contadores.

        Si hay score y umbral, se muestran al lado del estado para que el
        técnico vea cuán lejos está del umbral en cada disparo
        (p.ej. "OK  (score=3.8156 / umbral=4.1287)").
        """
        defecto = resultado.get("defecto")
        score   = resultado.get("scores", {}).get("color")
        umbral  = resultado.get("umbral")

        # Sufijo "score / umbral" — se añade solo cuando hay valores numéricos.
        sufijo = ""
        if score is not None and umbral is not None:
            sufijo = f"   (score={score:.4f} / umbral={umbral:.4f})"
        elif umbral is not None:
            sufijo = f"   (umbral={umbral:.4f})"

        if resultado["ok"]:
            self.lbl_resultado.setText(f"OK{sufijo}")
            self.lbl_resultado.setStyleSheet("""
                QLabel {
                    font-size: 24px; font-weight: bold;
                    color: white; background-color: #4CAF50;
                    border-radius: 8px; padding: 10px;
                }
            """)
            self.total_buenas += 1
        elif defecto in ("sin_luz", "sin_etiqueta"):
            # Motivo específico derivado del detalle del detector
            # (paleta distinta / etiqueta cortada / brillo / etc.).
            texto = _texto_motivo_rechazo(defecto, resultado.get("detalle", ""))
            self.lbl_resultado.setText(f"{texto}{sufijo}")
            self.lbl_resultado.setStyleSheet("""
                QLabel {
                    font-size: 22px; font-weight: bold;
                    color: white; background-color: #E65100;
                    border-radius: 8px; padding: 10px;
                }
            """)
            self.total_malas += 1
        else:
            # Defecto real detectado por el modelo de color (o un valor
            # inesperado — se muestra tal cual para diagnóstico).
            if defecto == "color":
                texto = "DEFECTO: MANCHA / COLOR"
            else:
                texto = f"DEFECTO: {defecto or 'desconocido'}"
            self.lbl_resultado.setText(f"{texto}{sufijo}")
            self.lbl_resultado.setStyleSheet("""
                QLabel {
                    font-size: 22px; font-weight: bold;
                    color: white; background-color: #f44336;
                    border-radius: 8px; padding: 10px;
                }
            """)
            self.total_malas += 1

        self._refrescar_conteo()

    def reiniciar_contadores(self):
        """Pone buenas/malas en cero y limpia el indicador."""
        self.total_buenas = 0
        self.total_malas  = 0
        self._refrescar_conteo()

        self.lbl_resultado.setText("ESPERANDO")
        self.lbl_resultado.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #888888;
                background-color: #333333;
                border-radius: 8px;
                padding: 10px;
            }
        """)

    def refrescar_referencias(self, nombres: list, activa: str = None):
        """
        Rellena el dropdown con la lista de referencias y marca la activa.

        Parámetros:
            nombres (list[str]): lista de nombres de referencias disponibles.
            activa  (str):       nombre de la activa (None si no hay ninguna).
        """
        self._bloquear_signal_combo = True
        try:
            self.combo_referencia.clear()

            if not nombres:
                self.combo_referencia.addItem("(sin referencias — crea una)")
                self.combo_referencia.setEnabled(False)
                return

            self.combo_referencia.setEnabled(True)
            self.combo_referencia.addItems(nombres)

            if activa and activa in nombres:
                self.combo_referencia.setCurrentIndex(nombres.index(activa))
        finally:
            self._bloquear_signal_combo = False

    # ── Handlers internos ──────────────────────────────────────────────────────

    def _al_cambiar_combo(self, indice: int):
        if self._bloquear_signal_combo:
            return
        if indice < 0:
            return
        nombre = self.combo_referencia.itemText(indice)
        if nombre and not nombre.startswith("("):
            self.sig_cambiar_referencia.emit(nombre)

    def _refrescar_conteo(self):
        total = self.total_buenas + self.total_malas
        self.lbl_conteo.setText(
            f"Buenas: {self.total_buenas}  |  "
            f"Malas: {self.total_malas}  |  "
            f"Total: {total}"
        )
