"""
VistaManual - Pantalla de captura manual (Prototipo)
======================================================
Permite al técnico tomar una foto sin pasar por la inspección, útil durante
la puesta en marcha para verificar el encuadre y la iluminación.

En el prototipo de etiquetas individuales no hay motor ni banda que mover,
así que esta vista sólo conserva el botón de captura.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from PyQt5.QtCore import Qt, pyqtSignal


class VistaManual(QWidget):
    """Vista de captura manual."""

    sig_captura            = pyqtSignal()
    sig_toggle_camara_viva = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._camara_viva = False
        self._crear_interfaz()

    def _crear_interfaz(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Título ──
        titulo = QLabel("Captura manual")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")

        # ── Área de imagen ──
        self.lbl_preview = QLabel("Sin imagen")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumHeight(300)
        self.lbl_preview.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e; color: #666666;
                font-size: 18px; border: 2px solid #333333;
                border-radius: 8px;
            }
        """)

        # ── Fila de botones ──
        fila_botones = QHBoxLayout()

        self.btn_abrir_camara = QPushButton("Abrir cámara")
        self.btn_abrir_camara.setMinimumHeight(60)
        self.btn_abrir_camara.setStyleSheet("""
            QPushButton {
                font-size: 18px; font-weight: bold;
                background-color: #4CAF50; color: white;
                border-radius: 10px; padding: 10px 20px;
            }
            QPushButton:pressed { background-color: #388E3C; }
        """)

        self.btn_capturar = QPushButton("Capturar foto")
        self.btn_capturar.setMinimumHeight(60)
        self.btn_capturar.setStyleSheet("""
            QPushButton {
                font-size: 18px; font-weight: bold;
                background-color: #2196F3; color: white;
                border-radius: 10px; padding: 10px 20px;
            }
            QPushButton:pressed { background-color: #1976D2; }
        """)

        fila_botones.addWidget(self.btn_abrir_camara)
        fila_botones.addWidget(self.btn_capturar)

        # ── Armar layout ──
        layout.addWidget(titulo)
        layout.addWidget(self.lbl_preview, stretch=3)
        layout.addLayout(fila_botones)

        # ── Conectar botones ──
        self.btn_capturar.clicked.connect(self.sig_captura.emit)
        self.btn_abrir_camara.clicked.connect(self._toggle_camara_viva)

    def _toggle_camara_viva(self):
        self._camara_viva = not self._camara_viva
        if self._camara_viva:
            self.btn_abrir_camara.setText("Cerrar cámara")
            self.btn_abrir_camara.setStyleSheet("""
                QPushButton {
                    font-size: 18px; font-weight: bold;
                    background-color: #f44336; color: white;
                    border-radius: 10px; padding: 10px 20px;
                }
                QPushButton:pressed { background-color: #c62828; }
            """)
        else:
            self.btn_abrir_camara.setText("Abrir cámara")
            self.btn_abrir_camara.setStyleSheet("""
                QPushButton {
                    font-size: 18px; font-weight: bold;
                    background-color: #4CAF50; color: white;
                    border-radius: 10px; padding: 10px 20px;
                }
                QPushButton:pressed { background-color: #388E3C; }
            """)
        self.sig_toggle_camara_viva.emit()

    def resetear_camara_viva(self):
        """Restablece el estado del botón a 'cerrado' sin emitir señal."""
        self._camara_viva = False
        self.btn_abrir_camara.setText("Abrir cámara")
        self.btn_abrir_camara.setStyleSheet("""
            QPushButton {
                font-size: 18px; font-weight: bold;
                background-color: #4CAF50; color: white;
                border-radius: 10px; padding: 10px 20px;
            }
            QPushButton:pressed { background-color: #388E3C; }
        """)
