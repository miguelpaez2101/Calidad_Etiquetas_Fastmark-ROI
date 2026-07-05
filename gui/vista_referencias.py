"""
VistaReferencias - Pantalla para elegir o crear referencia de etiqueta
========================================================================
Muestra la lista de referencias guardadas en `referencias/` y permite:
    - Ver el estado de cada una (cuántas imágenes en el banco, si tiene modelo)
    - Seleccionar una para hacerla activa
    - Crear una nueva referencia (pide nombre + descripción)
    - Eliminar una referencia (con confirmación)

MainWindow escucha las señales de esta vista y aplica los cambios.
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QTextEdit,
    QFormLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal


class VistaReferencias(QWidget):
    """
    Pantalla de gestión de referencias.

    Señales:
        sig_seleccionar(str)          — el operador pidió hacer activa una referencia
        sig_crear(str, str)           — crear nueva: (nombre, descripcion)
        sig_eliminar(str)             — borrar una referencia
    """

    sig_seleccionar = pyqtSignal(str)
    sig_crear       = pyqtSignal(str, str)
    sig_eliminar    = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._activa = None  # nombre de la referencia activa
        self._crear_interfaz()

    def _crear_interfaz(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── Título ──
        titulo = QLabel("Referencias de etiqueta")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")

        self.lbl_info = QLabel(
            "Cada referencia es un modelo de color entrenado con sus propias\n"
            "imágenes. Selecciona la que corresponde a la etiqueta que estás\n"
            "inspeccionando hoy."
        )
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: #888888; padding: 4px 10px;")

        # ── Lista ──
        self.lista = QListWidget()
        self.lista.setStyleSheet("""
            QListWidget {
                font-size: 15px; border: 2px solid #333333;
                border-radius: 8px; background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QListWidget::item {
                padding: 10px; border-bottom: 1px solid #2a2a3e;
            }
            QListWidget::item:selected {
                background-color: #2e4e6e; color: white;
            }
        """)
        self.lista.itemDoubleClicked.connect(self._al_doble_clic)

        # ── Botones ──
        botones = QHBoxLayout()

        self.btn_seleccionar = QPushButton("Seleccionar como activa")
        self.btn_seleccionar.setMinimumHeight(50)
        self.btn_seleccionar.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #4CAF50; color: white;
                border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #388E3C; }
            QPushButton:disabled { background-color: #555555; color: #999999; }
        """)

        self.btn_nueva = QPushButton("Nueva referencia")
        self.btn_nueva.setMinimumHeight(50)
        self.btn_nueva.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #0288D1; color: white;
                border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #01579B; }
        """)

        self.btn_eliminar = QPushButton("Eliminar")
        self.btn_eliminar.setMinimumHeight(50)
        self.btn_eliminar.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold;
                background-color: #D32F2F; color: white;
                border-radius: 8px; padding: 8px 16px;
            }
            QPushButton:pressed { background-color: #B71C1C; }
            QPushButton:disabled { background-color: #555555; color: #999999; }
        """)

        botones.addWidget(self.btn_seleccionar)
        botones.addWidget(self.btn_nueva)
        botones.addWidget(self.btn_eliminar)

        # ── Estado inferior ──
        self.lbl_estado = QLabel("")
        self.lbl_estado.setAlignment(Qt.AlignCenter)
        self.lbl_estado.setStyleSheet("font-size: 14px; color: #888888; padding: 5px;")

        # ── Armar ──
        layout.addWidget(titulo)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.lista, stretch=1)
        layout.addLayout(botones)
        layout.addWidget(self.lbl_estado)

        # ── Conectar ──
        self.btn_seleccionar.clicked.connect(self._al_seleccionar)
        self.btn_nueva.clicked.connect(self._al_crear_nueva)
        self.btn_eliminar.clicked.connect(self._al_eliminar)
        self.lista.itemSelectionChanged.connect(self._actualizar_habilitados)

        self._actualizar_habilitados()

    # ── API pública (MainWindow la llama) ──────────────────────────────────────

    def refrescar(self, referencias: list, activa: str = None):
        """
        Rellena la lista con las referencias del disco.

        Parámetros:
            referencias (list[tuple]): lista de (nombre, estado_texto) por cada
                referencia existente. Ej: [("Surfer_350SL", "15 imágenes · modelo entrenado")]
            activa (str): nombre de la referencia activa (se marca con ★).
        """
        self._activa = activa
        self.lista.clear()

        if not referencias:
            item = QListWidgetItem(
                "Sin referencias. Toca 'Nueva referencia' para crear la primera."
            )
            item.setFlags(Qt.NoItemFlags)  # no seleccionable
            self.lista.addItem(item)
            self.lbl_estado.setText("")
            self._actualizar_habilitados()
            return

        for nombre, estado in referencias:
            prefijo = "★ " if nombre == activa else "   "
            item = QListWidgetItem(f"{prefijo}{nombre}\n     {estado}")
            item.setData(Qt.UserRole, nombre)
            if nombre == activa:
                item.setForeground(Qt.green)
            self.lista.addItem(item)

        if activa:
            self.lbl_estado.setText(f"Activa: {activa}")
        else:
            self.lbl_estado.setText(
                "Ninguna referencia activa — selecciona una o crea la primera."
            )

        self._actualizar_habilitados()

    # ── Handlers internos ──────────────────────────────────────────────────────

    def _actualizar_habilitados(self):
        """Habilita/deshabilita botones según si hay selección."""
        nombre = self._nombre_seleccionado()
        self.btn_seleccionar.setEnabled(nombre is not None and nombre != self._activa)
        self.btn_eliminar.setEnabled(nombre is not None)

    def _nombre_seleccionado(self):
        items = self.lista.selectedItems()
        if not items:
            return None
        nombre = items[0].data(Qt.UserRole)
        return nombre  # None si el item es el placeholder "Sin referencias"

    def _al_seleccionar(self):
        nombre = self._nombre_seleccionado()
        if nombre:
            self.sig_seleccionar.emit(nombre)

    def _al_doble_clic(self, item):
        nombre = item.data(Qt.UserRole)
        if nombre and nombre != self._activa:
            self.sig_seleccionar.emit(nombre)

    def _al_eliminar(self):
        nombre = self._nombre_seleccionado()
        if not nombre:
            return

        respuesta = QMessageBox.warning(
            self, "Eliminar referencia",
            f"¿Eliminar la referencia '{nombre}'?\n\n"
            "Esto borra permanentemente:\n"
            "  • El modelo de color (.pkl)\n"
            "  • Todas las imágenes del banco\n"
            "  • La calibración de paleta y el umbral guardados\n\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        if respuesta == QMessageBox.Yes:
            self.sig_eliminar.emit(nombre)

    def _al_crear_nueva(self):
        dialogo = _DialogoNuevaReferencia(self)
        if dialogo.exec_() == QDialog.Accepted:
            nombre, descripcion = dialogo.resultado()
            self.sig_crear.emit(nombre, descripcion)


# ── Diálogo interno para crear una referencia nueva ──────────────────────────

class _DialogoNuevaReferencia(QDialog):
    """Pide nombre y descripción para la nueva referencia."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nueva referencia de etiqueta")
        self.setMinimumWidth(400)
        self._crear_interfaz()

    def _crear_interfaz(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Crea una nueva referencia (un nuevo modelo entrenado).\n"
            "Después tendrás que agregarle imágenes y entrenarla desde Mantenimiento."
        )
        info.setStyleSheet("color: #888888; padding: 4px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.edit_nombre = QLineEdit()
        self.edit_nombre.setPlaceholderText("Ej: Surfer_350SL, Glifosato_1L")
        self.edit_nombre.setToolTip(
            "Solo letras (sin tildes), números, guion (-) y guion bajo (_).\n"
            "Sin espacios ni caracteres especiales."
        )
        form.addRow("Nombre:", self.edit_nombre)

        self.edit_descripcion = QTextEdit()
        self.edit_descripcion.setPlaceholderText(
            "Texto libre para reconocer esta etiqueta (opcional)"
        )
        self.edit_descripcion.setMaximumHeight(80)
        form.addRow("Descripción:", self.edit_descripcion)

        layout.addLayout(form)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

        self.edit_nombre.setFocus()

    def resultado(self) -> tuple:
        return (
            self.edit_nombre.text().strip(),
            self.edit_descripcion.toPlainText().strip(),
        )
