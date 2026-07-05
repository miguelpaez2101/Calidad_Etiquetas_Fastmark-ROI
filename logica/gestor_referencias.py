"""
GestorReferencias - Administra las referencias de etiqueta del sistema
========================================================================
Cada referencia (ej: Surfer_350SL, Glifosato_1L, etc.) es un modelo entrenado
independiente con su propia calibración de paleta, banco de imágenes y
umbral.  (El recorte de cámara es GLOBAL — vive en config/calibracion.json.)

Estructura en disco:
    referencias/
    ├── Surfer_350SL/
    │   ├── meta.json           ← descripción, umbral, fechas
    │   ├── modelo_roi.pkl      ← modelo entrenado (puede no existir aún)
    │   └── buenas/             ← banco de imágenes para entrenar
    │       ├── buena_001.jpg
    │       └── ...
    └── Otra_Referencia/
        └── ...

¿Por qué una carpeta por referencia?
    Cada etiqueta tiene apariencia distinta → su modelo de color no sirve para
    otra. Aislando todo por carpeta, cambiar de referencia es solo cargar otro
    .pkl y otra calibración — sin tocar la configuración de la otra.
"""

import os
import json
import shutil
import logging
import re
from datetime import datetime


# Ruta base del proyecto (la carpeta padre de logica/)
_RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_REFERENCIAS = os.path.join(_RUTA_BASE, "referencias")


# Nombres válidos para referencias: letras, dígitos, _ y - (sin espacios ni tildes
# porque se usan como nombres de carpeta y en paths del .pkl).
_REGEX_NOMBRE_VALIDO = re.compile(r"^[A-Za-z0-9_\-]+$")


class GestorReferencias:
    """
    CRUD de referencias en `referencias/`. No mantiene estado — cada llamada
    lee o escribe del disco directamente (así no hace falta recargar cuando
    otra parte del programa agrega un archivo).
    """

    # ── Consultas ──────────────────────────────────────────────────────────────

    def listar(self) -> list:
        """
        Retorna la lista de nombres de referencias ordenada alfabéticamente.
        Si la carpeta raíz no existe, retorna lista vacía.
        """
        if not os.path.isdir(RUTA_REFERENCIAS):
            return []

        nombres = []
        for entrada in os.listdir(RUTA_REFERENCIAS):
            ruta = os.path.join(RUTA_REFERENCIAS, entrada)
            # Solo carpetas que tengan meta.json son referencias válidas
            if os.path.isdir(ruta) and os.path.exists(os.path.join(ruta, "meta.json")):
                nombres.append(entrada)

        nombres.sort(key=str.lower)
        return nombres

    def existe(self, nombre: str) -> bool:
        """True si existe una carpeta de referencia con ese nombre."""
        ruta = self.ruta_carpeta(nombre)
        return os.path.isdir(ruta) and os.path.exists(os.path.join(ruta, "meta.json"))

    def validar_nombre(self, nombre: str) -> tuple:
        """
        Verifica que el nombre sea válido para ser una carpeta y único.

        Retorna:
            (valido: bool, mensaje_error: str)
        """
        if not nombre or not nombre.strip():
            return False, "El nombre no puede estar vacío."

        nombre = nombre.strip()

        if not _REGEX_NOMBRE_VALIDO.match(nombre):
            return False, (
                "El nombre solo puede contener letras (sin tildes), números, "
                "guion (-) y guion bajo (_). Sin espacios ni caracteres especiales."
            )

        if len(nombre) > 50:
            return False, "El nombre es demasiado largo (máximo 50 caracteres)."

        if self.existe(nombre):
            return False, f"Ya existe una referencia llamada '{nombre}'."

        return True, ""

    # ── Rutas ──────────────────────────────────────────────────────────────────

    def ruta_carpeta(self, nombre: str) -> str:
        return os.path.join(RUTA_REFERENCIAS, nombre)

    def ruta_meta(self, nombre: str) -> str:
        return os.path.join(self.ruta_carpeta(nombre), "meta.json")

    def ruta_modelo(self, nombre: str) -> str:
        return os.path.join(self.ruta_carpeta(nombre), "modelo_roi.pkl")

    def ruta_banco(self, nombre: str) -> str:
        return os.path.join(self.ruta_carpeta(nombre), "buenas")

    # ── Meta (descripción, recorte, umbral, fechas) ────────────────────────────

    def cargar_meta(self, nombre: str) -> dict:
        """
        Lee el meta.json de una referencia.

        Retorna dict con las claves:
            nombre (str)
            descripcion (str)
            creada (str ISO 8601)
            recorte (dict {x,y,w,h})
            umbral_modelo (float|None)

        Si el archivo no existe o está corrupto, retorna un meta vacío.
        """
        ruta = self.ruta_meta(nombre)
        if not os.path.exists(ruta):
            return self._meta_vacio(nombre)

        try:
            with open(ruta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"GestorReferencias: meta.json corrupto en '{ruta}': {e}")
            return self._meta_vacio(nombre)

        # Rellenar campos faltantes con defaults
        defaults = self._meta_vacio(nombre)
        for clave, valor in defaults.items():
            meta.setdefault(clave, valor)
        return meta

    def guardar_meta(self, nombre: str, meta: dict):
        """
        Escribe meta.json de forma atómica (tmp + rename).
        """
        ruta = self.ruta_meta(nombre)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        ruta_tmp = ruta + ".tmp"

        with open(ruta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(ruta_tmp, ruta)

    # ── Operaciones ────────────────────────────────────────────────────────────

    def crear(self, nombre: str, descripcion: str = "") -> tuple:
        """
        Crea una nueva referencia: la carpeta, meta.json, subcarpeta buenas/.

        Parámetros:
            nombre (str): Nombre único sin espacios/tildes.
            descripcion (str): Texto libre para reconocer la referencia.

        Retorna:
            (exito: bool, mensaje: str)
        """
        valido, msg = self.validar_nombre(nombre)
        if not valido:
            return False, msg

        nombre = nombre.strip()
        carpeta = self.ruta_carpeta(nombre)
        banco   = self.ruta_banco(nombre)

        try:
            os.makedirs(carpeta, exist_ok=True)
            os.makedirs(banco,   exist_ok=True)
        except OSError as e:
            return False, f"No se pudo crear la carpeta: {e}"

        meta = self._meta_vacio(nombre)
        meta["descripcion"] = descripcion.strip()
        meta["creada"]      = datetime.now().isoformat(timespec="seconds")

        try:
            self.guardar_meta(nombre, meta)
        except OSError as e:
            return False, f"No se pudo escribir meta.json: {e}"

        logging.info(f"GestorReferencias: referencia '{nombre}' creada.")
        return True, f"Referencia '{nombre}' creada."

    def eliminar(self, nombre: str) -> tuple:
        """
        Borra por completo la carpeta de una referencia (modelo, banco, meta).

        Retorna:
            (exito: bool, mensaje: str)
        """
        if not self.existe(nombre):
            return False, f"La referencia '{nombre}' no existe."

        carpeta = self.ruta_carpeta(nombre)
        try:
            shutil.rmtree(carpeta)
        except OSError as e:
            return False, f"No se pudo eliminar la carpeta: {e}"

        logging.info(f"GestorReferencias: referencia '{nombre}' eliminada.")
        return True, f"Referencia '{nombre}' eliminada."

    # ── Estado (para la UI) ────────────────────────────────────────────────────

    def tiene_modelo(self, nombre: str) -> bool:
        """True si existe modelo_roi.pkl en la carpeta de la referencia."""
        return os.path.exists(self.ruta_modelo(nombre))

    def contar_imagenes_banco(self, nombre: str) -> int:
        """Cuenta cuántas imágenes .jpg/.jpeg/.png hay en el banco."""
        banco = self.ruta_banco(nombre)
        if not os.path.isdir(banco):
            return 0
        return sum(
            1 for f in os.listdir(banco)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def resumen_estado(self, nombre: str) -> str:
        """
        Descripción corta del estado de una referencia para mostrar en la UI.
        Ej: "15 imágenes · modelo entrenado"
             "3 imágenes · sin modelo"
        """
        n      = self.contar_imagenes_banco(nombre)
        modelo = "modelo entrenado" if self.tiene_modelo(nombre) else "sin modelo"
        return f"{n} imágenes · {modelo}"

    # ── Privados ───────────────────────────────────────────────────────────────

    @staticmethod
    def _meta_vacio(nombre: str) -> dict:
        """
        Plantilla de meta.json con valores por defecto.

        El recorte vive ahora en calibracion.json (global) — no se duplica
        aquí.  Sólo guardamos lo que es ESTRICTAMENTE por-referencia.
        """
        return {
            "version":       "2.0",
            "nombre":        nombre,
            "descripcion":   "",
            "creada":        "",
            "umbral_modelo": None,
        }
