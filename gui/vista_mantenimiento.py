"""
VistaMantenimiento - Pantalla de calibración (protegida con contraseña)
========================================================================
Solo para técnicos/ingenieros. Permite:
    - Definir el área de recorte de la cámara (zona útil de la etiqueta)
    - Capturar imágenes y probar la inspección
    - Agregar imágenes al banco de entrenamiento del modelo de color
    - Entrenar el modelo de color (estadísticas HSV por celda)
    - Ajustar el umbral del modelo de color
    - Guardar la configuración

El detector por celdas (DetectorROI) aprende la apariencia de las
etiquetas buenas del banco y rechaza manchas, oclusiones y desvíos de
color localizados.  Las ROIs de la calibración solo dan nombre a las
zonas en los mensajes de defecto — la detección no depende de ellas.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QDoubleSpinBox,
    QFrame,
)
from PyQt5.QtCore import Qt, QLocale


class VistaMantenimiento(QWidget):
    """Vista de mantenimiento y calibración."""

    def __init__(self):
        super().__init__()
        self._crear_interfaz()

    def _crear_interfaz(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Título ──
        titulo = QLabel("Mantenimiento y calibración")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")

        # ── Banner de referencia activa ──
        # Todo lo que se hace en esta pantalla (recorte, banco, modelo, umbral)
        # queda guardado dentro de la referencia activa. Mostrarlo a la vista
        # evita confusiones tipo "¿por qué se entrenó otro modelo?".
        self.lbl_ref_activa = QLabel("Referencia activa: (ninguna)")
        self.lbl_ref_activa.setAlignment(Qt.AlignCenter)
        self.lbl_ref_activa.setStyleSheet("""
            QLabel {
                font-size: 15px; font-weight: bold;
                color: #E8F5E9;
                background-color: #2E7D32;
                border-radius: 6px;
                padding: 8px 14px;
            }
        """)

        # ── Área de imagen ──
        # Ocupa el lado izquierdo en el layout nuevo (columna amplia) para
        # que la vista rectificada (canvas canónico) se aprecie en detalle.
        self.lbl_imagen = QLabel("Capture una imagen para comenzar la calibración")
        self.lbl_imagen.setAlignment(Qt.AlignCenter)
        self.lbl_imagen.setMinimumHeight(420)
        self.lbl_imagen.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e; color: #666666;
                font-size: 16px; border: 2px solid #333333;
                border-radius: 8px;
            }
        """)

        # ── Panel de controles (columna derecha) ──────────────────────────────
        # Todos los botones vertical en un ancho fijo para que no compitan con
        # la imagen por el espacio vertical.
        panel = QWidget()
        panel.setFixedWidth(330)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        # --- Captura / calibración ---
        # Paso 1 (debe hacerse ANTES del banco): calibrar la referencia.
        # Descubre la paleta de colores (k-means HSV) sobre una maestra
        # rectificada y guarda las ROIs con nombre.  El entrenamiento queda
        # bloqueado en MainWindow hasta que la calibración exista.
        self.btn_calibrar_referencia = QPushButton("Calibrar referencia")
        self.btn_calibrar_referencia.setMinimumHeight(52)
        self.btn_calibrar_referencia.setToolTip(
            "Detecta la paleta de colores de la etiqueta y nombra sus zonas.\n"
            "Es OBLIGATORIO antes de capturar el banco — sin esto el sistema\n"
            "no sabe qué colores esperar para esta referencia."
        )
        self.btn_calibrar_referencia.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #5E35B1;
                color: white; border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #311B92; }
        """)

        self.btn_definir_recorte = QPushButton("Definir área de inspección")
        self.btn_definir_recorte.setMinimumHeight(52)
        self.btn_definir_recorte.setToolTip(
            "Recorta la imagen de la cámara para que solo abarque la etiqueta.\n"
            "Hacer este paso ANTES de capturar imágenes de entrenamiento."
        )
        self.btn_definir_recorte.setStyleSheet("""
            QPushButton {
                font-size: 15px; background-color: #00796B;
                color: white; border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #004D40; }
        """)

        self.btn_capturar = QPushButton("Capturar imagen")
        self.btn_capturar.setMinimumHeight(52)
        self.btn_capturar.setStyleSheet("""
            QPushButton {
                font-size: 15px; background-color: #607D8B;
                color: white; border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #455A64; }
        """)

        self.btn_probar = QPushButton("Probar inspección")
        self.btn_probar.setMinimumHeight(52)
        self.btn_probar.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #9C27B0; color: white;
                border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #6A1B9A; }
        """)

        self.btn_guardar_foto = QPushButton("Guardar foto de prueba")
        self.btn_guardar_foto.setMinimumHeight(52)
        self.btn_guardar_foto.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                background-color: #0277BD; color: white;
                border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #01579B; }
        """)

        col.addWidget(self.btn_calibrar_referencia)
        col.addWidget(self.btn_definir_recorte)
        col.addWidget(self.btn_capturar)
        col.addWidget(self.btn_probar)
        col.addWidget(self.btn_guardar_foto)

        # Separador visual antes del grupo del modelo de color.
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333333; background-color: #333333;")
        col.addWidget(sep)

        # --- Grupo modelo de color (adaptado a columna angosta) ---
        grupo_modelo = QGroupBox("Modelo de color — manchas y desvíos")
        grupo_modelo.setStyleSheet("""
            QGroupBox {
                font-size: 13px; font-weight: bold;
                border: 1px solid #444444; border-radius: 6px;
                margin-top: 6px; padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
            }
        """)
        col_modelo = QVBoxLayout(grupo_modelo)
        col_modelo.setSpacing(6)

        self.btn_agregar_entrenamiento = QPushButton("Agregar imagen al banco")
        self.btn_agregar_entrenamiento.setMinimumHeight(46)
        self.btn_agregar_entrenamiento.setToolTip(
            "Guarda el frame actual como imagen buena de entrenamiento.\n"
            "Agrega 10–30 imágenes antes de entrenar el modelo."
        )
        self.btn_agregar_entrenamiento.setStyleSheet("""
            QPushButton {
                font-size: 14px; background-color: #0288D1;
                color: white; border-radius: 8px; padding: 6px 14px;
            }
            QPushButton:pressed { background-color: #01579B; }
        """)

        self.btn_entrenar = QPushButton("Entrenar modelo de color")
        self.btn_entrenar.setMinimumHeight(46)
        self.btn_entrenar.setToolTip(
            "Entrena el modelo con las imágenes del banco.\n"
            "Es casi instantáneo (menos de un segundo)."
        )
        self.btn_entrenar.setStyleSheet("""
            QPushButton {
                font-size: 14px; font-weight: bold;
                background-color: #E65100; color: white;
                border-radius: 8px; padding: 6px 14px;
            }
            QPushButton:pressed { background-color: #BF360C; }
        """)

        fila_umbral = QHBoxLayout()
        lbl_umbral = QLabel("Umbral:")
        lbl_umbral.setFixedWidth(62)

        self.spin_umbral_pc = QDoubleSpinBox()
        # Locale C para forzar "." como separador decimal.  Con el locale del
        # sistema en español, Qt espera "," y rechaza silenciosamente "4.12"
        # (el valor tecleado se revierte al anterior y parece que "no hace nada").
        self.spin_umbral_pc.setLocale(QLocale.c())
        self.spin_umbral_pc.setMinimum(0.001)
        self.spin_umbral_pc.setMaximum(99.0)
        self.spin_umbral_pc.setSingleStep(0.01)
        self.spin_umbral_pc.setDecimals(4)
        self.spin_umbral_pc.setValue(0.050)
        # keyboardTracking=False: valueChanged dispara solo al presionar Enter
        # o al salir del foco.  Evita que cada tecla intermedia emita un
        # valor parcial (p.ej. "4" antes de escribir ".12") que sobreescriba
        # el umbral del detector con un número equivocado.
        self.spin_umbral_pc.setKeyboardTracking(False)
        self.spin_umbral_pc.setFixedWidth(110)
        self.spin_umbral_pc.setToolTip(
            "Score máximo para considerar la etiqueta OK.\n"
            "Se calcula automáticamente al entrenar; ajustar si hay falsos positivos.\n"
            "Separador decimal: punto (.).  Presiona Enter para aplicar."
        )
        fila_umbral.addWidget(lbl_umbral)
        fila_umbral.addWidget(self.spin_umbral_pc)
        fila_umbral.addStretch()

        self.lbl_banco_estado = QLabel("Banco: sin imágenes")
        self.lbl_banco_estado.setStyleSheet("font-size: 12px; color: #888888;")
        self.lbl_banco_estado.setWordWrap(True)

        col_modelo.addWidget(self.btn_agregar_entrenamiento)
        col_modelo.addWidget(self.btn_entrenar)
        col_modelo.addLayout(fila_umbral)
        col_modelo.addWidget(self.lbl_banco_estado)

        col.addWidget(grupo_modelo)
        col.addStretch()

        # ── Contenido principal: imagen + panel derecho ────────────────────────
        contenido = QHBoxLayout()
        contenido.setSpacing(12)
        contenido.addWidget(self.lbl_imagen, stretch=1)
        contenido.addWidget(panel)

        # ── Resultado de prueba ──
        self.lbl_resultado_prueba = QLabel("")
        self.lbl_resultado_prueba.setAlignment(Qt.AlignCenter)
        self.lbl_resultado_prueba.setMinimumHeight(40)
        self.lbl_resultado_prueba.setStyleSheet("font-size: 15px; border-radius: 6px;")

        # ── Estado ──
        self.lbl_estado = QLabel("Estado: sin calibrar")
        self.lbl_estado.setAlignment(Qt.AlignCenter)
        self.lbl_estado.setStyleSheet("font-size: 14px; color: #888888; padding: 5px;")

        # ── Armar layout ──
        layout.addWidget(titulo)
        layout.addWidget(self.lbl_ref_activa)
        layout.addLayout(contenido, stretch=1)
        layout.addWidget(self.lbl_resultado_prueba)
        layout.addWidget(self.lbl_estado)

        # ── IMPORTANTE ──
        # No conectamos los botones a slots internos aquí. MainWindow es el
        # único responsable de conectarlos — así evitamos que cada click
        # dispare dos handlers (uno dummy local + el real de MainWindow).

    # ── API pública (MainWindow la llama al cambiar de referencia) ────────────

    def actualizar_referencia_activa(self, nombre: str = None, estado: str = None):
        """
        Refresca el banner superior con la referencia activa.

        Parámetros:
            nombre: nombre de la referencia activa, o None si no hay ninguna.
            estado: texto corto tipo "15 imágenes · modelo entrenado" (opcional).
        """
        if not nombre:
            self.lbl_ref_activa.setText(
                "⚠ Sin referencia activa — selecciona o crea una en 'Referencias'"
            )
            self.lbl_ref_activa.setStyleSheet("""
                QLabel {
                    font-size: 15px; font-weight: bold;
                    color: white;
                    background-color: #C62828;
                    border-radius: 6px;
                    padding: 8px 14px;
                }
            """)
            return

        texto = f"Referencia activa: {nombre}"
        if estado:
            texto += f"   ·   {estado}"
        self.lbl_ref_activa.setText(texto)
        self.lbl_ref_activa.setStyleSheet("""
            QLabel {
                font-size: 15px; font-weight: bold;
                color: #E8F5E9;
                background-color: #2E7D32;
                border-radius: 6px;
                padding: 8px 14px;
            }
        """)

    # ── Detalle de inspección (popup) ─────────────────────────────────────────

    def mostrar_detalle_prueba(self, resultado: dict):
        """
        Muestra un popup con el resultado detallado de la inspección.

        Parámetros:
            resultado: dict devuelto por Inspector.inspeccionar_diagnostico()
                {
                    "ok":            bool,
                    "disponible":    bool,
                    "score":         float|None,
                    "umbral":        float,
                    "defecto":       str|None,
                    "detalle":       str,
                    "mapa_anomalia": np.ndarray|None
                }
        """
        C_OK  = "#1B5E20"
        C_MAL = "#B71C1C"
        C_AVS = "#555555"

        defecto = resultado.get("defecto")

        # El título del popup depende de dónde rechazó el pipeline.  Si la
        # imagen llegó al detector por celdas el título menciona el modelo
        # de color; si fue rechazada antes (brillo, presencia, rectificación),
        # usamos un título neutral para no atribuir el rechazo al modelo.
        if defecto in ("sin_luz", "sin_etiqueta"):
            titulo = "<b>Pre-check — detección de etiqueta</b>"
        else:
            titulo = "<b>Modelo de color — manchas y desvíos</b>"

        filas = []
        filas.append(
            f"<tr><td colspan='4' style='padding:6px 0 2px 0;'>{titulo}</td></tr>"
        )

        if defecto == "sin_luz":
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_MAL}; font-weight:bold;'>"
                f"Imagen demasiado oscura — luz apagada o sin etiqueta</td></tr>"
            )
        elif defecto == "sin_etiqueta":
            detalle = resultado.get("detalle",
                                    "La imagen no coincide con el banco aprendido.")
            # El subtítulo se deriva del detalle — cada razón que produce el
            # detector tiene una palabra clave estable que identifica el
            # motivo específico del rechazo.  Así el técnico ve de un vistazo
            # si debe ajustar iluminación, recorte, o cambiar la etiqueta.
            d = detalle.lower()
            if "paleta" in d:
                subtitulo = "Paleta de colores distinta"
            elif "fuera del área" in d or "cortada" in d:
                subtitulo = "Cortada por el borde del domo"
            elif "brillo" in d:
                subtitulo = "Brillo fuera del rango aprendido"
            elif "segmentar" in d or "contorno" in d or "pequeña" in d:
                subtitulo = "Sin etiqueta — no detectada en el domo"
            else:
                subtitulo = "Firma de presencia fuera de rango"
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_MAL}; font-weight:bold;'>"
                f"{subtitulo}</td></tr>"
            )
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_AVS};'>"
                f"<i>{detalle}</i></td></tr>"
            )
        elif defecto == "color":
            # El detalle ya trae las zonas afectadas y el score/umbral
            # (lo arma DetectorROI a partir de las celdas disparadas).
            detalle = resultado.get("detalle",
                                    "Color fuera del rango aprendido.")
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_MAL}; font-weight:bold;'>"
                f"Mancha o color fuera de rango — ver zonas en el detalle</td></tr>"
            )
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_AVS};'>"
                f"<i>{detalle}</i></td></tr>"
            )
        elif resultado.get("disponible", False) and resultado.get("score") is not None:
            score  = resultado.get("score")
            umbral = resultado.get("umbral", 0.0)
            ok     = resultado.get("ok", False)
            color  = C_OK if ok else C_MAL
            icono  = "PASA ✓" if ok else "NO PASA ✗"

            filas.append(
                f"<tr>"
                f"<td style='padding:2px 10px 2px 18px;'>Score color</td>"
                f"<td style='padding:2px 6px; color:{color}; font-weight:bold;'>"
                f"{score:.4f}</td>"
                f"<td style='padding:2px 6px; color:{C_AVS};'>"
                f"umbral {umbral:.4f}</td>"
                f"<td style='padding:2px 6px; color:{color}; font-weight:bold;'>"
                f"{icono}</td>"
                f"</tr>"
            )
        else:
            filas.append(
                f"<tr><td colspan='4' style='padding:2px 18px; color:{C_AVS};'>"
                f"<i>Sin modelo entrenado — capturar buenas y entrenar "
                f"el modelo de color</i></td></tr>"
            )

        html = "<table style='font-size:13px; border-collapse:collapse;'>"
        html += "".join(filas)
        html += "</table>"

        dialogo = QDialog(self)
        dialogo.setWindowTitle("Detalle de la inspección")
        dialogo.setMinimumWidth(420)

        layout_d = QVBoxLayout(dialogo)
        lbl = QLabel(html)
        lbl.setTextFormat(Qt.RichText)
        layout_d.addWidget(lbl)

        botones = QDialogButtonBox(QDialogButtonBox.Ok)
        botones.accepted.connect(dialogo.accept)
        layout_d.addWidget(botones)

        dialogo.exec_()
