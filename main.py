"""
Inspector de Etiquetas Individuales — Prototipo
===================================================
Este es el archivo principal. Al ejecutarlo, arranca la aplicación completa.

Para ejecutar:
    python3 main.py

¿Qué hace este archivo?
1. Configura el logging (consola + archivo rotatorio en logs/)
2. Crea la "aplicación" de Qt (necesaria para que funcionen las ventanas)
3. Crea la ventana principal (MainWindow)
4. La muestra en pantalla completa (para la pantalla táctil de 7")
5. Espera a que el usuario cierre la aplicación
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

# En la Jetson (pantalla táctil de 7") se usa pantalla completa.
# En Windows o WSL (laptops de desarrollo) se maximiza para no bloquear el escritorio.
_ES_WSL = "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False
_PANTALLA_COMPLETA = sys.platform != "win32" and not _ES_WSL

_RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
_RUTA_LOGS = os.path.join(_RUTA_BASE, "logs")


def _configurar_logging():
    """
    Configura el sistema de logs:
      - Salida a consola (para desarrollo)
      - Archivo rotatorio en logs/surfer.log (5 MB por archivo, 3 backups)

    El archivo rotatorio evita que el log crezca sin límite en producción.
    """
    os.makedirs(_RUTA_LOGS, exist_ok=True)

    formato = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler_consola = logging.StreamHandler()
    handler_consola.setFormatter(formato)

    handler_archivo = RotatingFileHandler(
        os.path.join(_RUTA_LOGS, "prototipo.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    handler_archivo.setFormatter(formato)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Limpiar handlers previos (por si algo más los configuró antes)
    root.handlers.clear()
    root.addHandler(handler_consola)
    root.addHandler(handler_archivo)

    logging.info("=" * 60)
    logging.info("Inspector de Etiquetas Individuales (Prototipo) — arranque")
    logging.info("=" * 60)


def main():
    _configurar_logging()

    # Paso 1: Crear la aplicación Qt
    # Solo puede existir UNA QApplication por programa.
    app = QApplication(sys.argv)

    # Paso 2: Crear la ventana principal
    ventana = MainWindow()

    # Paso 3: Mostrar la ventana
    # En la Jetson (pantalla táctil 7"): pantalla completa.
    # En Windows (desarrollo): maximizada.
    if _PANTALLA_COMPLETA:
        ventana.showFullScreen()
    else:
        ventana.showMaximized()

    # Paso 4: Ejecutar el loop de Qt
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
