"""
Configuracion - Manejo del archivo calibracion.json (Prototipo)
================================================================
Esta clase es la única responsable de leer y escribir la configuración global
del prototipo de inspección de etiquetas individuales.

¿Qué guarda calibracion.json?
    - Parámetros de cámara (device_id, resolución, fps)
    - Recorte de cámara global — la zona del frame donde cabe cualquier
      etiqueta cuando entra al domo.  Se calibra una sola vez al instalar
      el equipo y todas las referencias lo comparten (antes era por-
      referencia en meta.json).
    - Nombre de la referencia activa (qué etiqueta estamos inspeccionando hoy)
    - Contraseña hasheada con PBKDF2 + sal
    - Umbral de manchas (parámetro global de inspección)

Lo que NO guarda (es por referencia, en referencias/<nombre>/meta.json):
    - Umbral del modelo de color (umbral_modelo — varía por modelo entrenado)
    - Calibración de color (calibracion_color.json) — paleta detectada de
      la maestra de esa referencia.

Nota: el prototipo omite toda la configuración de motor/sensor IR. La captura
es manual (el operador presiona un botón) y no hay control de banda ni
trigger óptico en esta etapa.

Uso:
    config = Configuracion()
    config.cargar()
    nombre = config.referencia_activa
    config.referencia_activa = "Harvanta_50SL"
    config.guardar()
"""

import json
import hashlib
import os
import secrets
import logging


# Ruta base del proyecto (la carpeta que contiene main.py)
RUTA_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CONFIG = os.path.join(RUTA_BASE, "config", "calibracion.json")

# Parámetros de PBKDF2 para hash de contraseña
_PBKDF2_ITERACIONES = 200_000
_PBKDF2_HASH       = "sha256"
_PBKDF2_LARGO_SAL  = 16  # bytes


class Configuracion:
    """
    Lee y escribe la configuración del sistema en calibracion.json.
    """

    def __init__(self):
        self._datos    = {}
        self._cargada  = False

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    def cargar(self):
        """Lee calibracion.json y carga todos los valores en memoria."""
        if not os.path.exists(RUTA_CONFIG):
            logging.warning(
                f"Configuracion: no se encontró {RUTA_CONFIG}. "
                "Usando valores por defecto."
            )
            self._datos = self._valores_por_defecto()
            self._cargada = True
            return

        with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
            self._datos = json.load(f)

        logging.info(f"Configuracion: cargada desde {RUTA_CONFIG}.")
        self._cargada = True

    def guardar(self):
        """Escribe calibracion.json de forma atómica (tmp + os.replace)."""
        os.makedirs(os.path.dirname(RUTA_CONFIG), exist_ok=True)
        ruta_tmp = RUTA_CONFIG + ".tmp"

        with open(ruta_tmp, "w", encoding="utf-8") as f:
            json.dump(self._datos, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(ruta_tmp, RUTA_CONFIG)
        logging.info(f"Configuracion: guardada en {RUTA_CONFIG}.")

    # ── Contraseña ─────────────────────────────────────────────────────────────

    def verificar_contrasena(self, texto: str) -> bool:
        """Verifica contraseña PBKDF2-HMAC-SHA256. Sin contraseña → acceso libre."""
        datos_pw = self._datos.get("contrasena", {})
        hash_guardado = datos_pw.get("hash", "")
        sal_guardada  = datos_pw.get("sal",  "")

        if not hash_guardado or not sal_guardada:
            return True

        sal_bytes = bytes.fromhex(sal_guardada)
        hash_calculado = hashlib.pbkdf2_hmac(
            _PBKDF2_HASH,
            texto.encode("utf-8"),
            sal_bytes,
            _PBKDF2_ITERACIONES
        ).hex()

        return secrets.compare_digest(hash_calculado, hash_guardado)

    def establecer_contrasena(self, texto: str):
        """Guarda una nueva contraseña (PBKDF2 + sal aleatoria)."""
        sal = secrets.token_bytes(_PBKDF2_LARGO_SAL)
        hash_bytes = hashlib.pbkdf2_hmac(
            _PBKDF2_HASH,
            texto.encode("utf-8"),
            sal,
            _PBKDF2_ITERACIONES
        )
        self._datos["contrasena"] = {
            "hash": hash_bytes.hex(),
            "sal":  sal.hex(),
        }
        logging.info("Configuracion: contraseña actualizada.")

    # ── Referencia activa ──────────────────────────────────────────────────────

    @property
    def referencia_activa(self):
        """Nombre de la referencia activa, o None si no hay ninguna."""
        return self._datos.get("referencia_activa") or None

    @referencia_activa.setter
    def referencia_activa(self, nombre):
        if nombre is None:
            self._datos.pop("referencia_activa", None)
        else:
            self._datos["referencia_activa"] = str(nombre)
        logging.info(f"Configuracion: referencia activa = {nombre}")

    # ── Umbral de manchas (para DetectorManchas futuro) ────────────────────────

    @property
    def umbral_manchas(self) -> int:
        return self._datos.get("inspeccion", {}).get("umbral_manchas", 100)

    @umbral_manchas.setter
    def umbral_manchas(self, valor: int):
        inspeccion = self._datos.setdefault("inspeccion", {})
        inspeccion["umbral_manchas"] = max(1, int(valor))

    # ── Secciones de hardware ──────────────────────────────────────────────────

    @property
    def config_camara(self) -> dict:
        """Sección 'camara' del JSON (device_id, resolución, fps)."""
        return self._datos.get("camara", {})

    # ── Recorte global ─────────────────────────────────────────────────────────

    @property
    def recorte(self) -> dict:
        """
        Recorte global de cámara aplicado a todas las referencias.

        Devuelve dict {x, y, w, h}.  Si no existe (instalación nueva o
        recién migrada), retorna {0, 0, 0, 0} — el caller interpreta
        eso como "sin recorte".
        """
        rec = self._datos.get("recorte", {})
        return {
            "x": int(rec.get("x", 0)),
            "y": int(rec.get("y", 0)),
            "w": int(rec.get("w", 0)),
            "h": int(rec.get("h", 0)),
        }

    @recorte.setter
    def recorte(self, valor: dict):
        self._datos["recorte"] = {
            "x": int(valor.get("x", 0)),
            "y": int(valor.get("y", 0)),
            "w": int(valor.get("w", 0)),
            "h": int(valor.get("h", 0)),
        }

    # ── Valores por defecto ────────────────────────────────────────────────────

    def _valores_por_defecto(self) -> dict:
        """Estructura mínima del JSON cuando no existe el archivo."""
        return {
            "version":    "3.0",
            "contrasena": {"hash": "", "sal": ""},
            "camara":     {"device_id": 0, "resolucion": [4000, 3000], "fps": 12},
            "recorte":    {"x": 0, "y": 0, "w": 0, "h": 0},
            "referencia_activa": None,
            "inspeccion": {"umbral_manchas": 100},
        }
