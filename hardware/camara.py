"""
Camara - Clase que encapsula la cámara ELP USB 48MP
====================================================
Modelo: ELP-USB48MP02-CFV(3.6-10)
Sensor: IMX586 — 48 megapíxeles
Interfaz: USB 2.0
Resolución de trabajo: 4000 × 3000 px @ 12 fps en formato MJPEG

¿Por qué encapsular la cámara en su propia clase?
    OpenCV accede a la cámara con cv2.VideoCapture. Si lo usamos directo
    en muchos lugares del código, cualquier cambio (número de cámara, resolución,
    formato) requiere modificar múltiples archivos. Al tener una clase,
    cambiamos un solo lugar.

Modo simulación:
    Si no hay cámara conectada (por ejemplo, en desarrollo en otra PC),
    la clase devuelve imágenes sintéticas de prueba en lugar de fallar.

Formato MJPEG:
    La cámara transmite los frames comprimidos en JPEG. Esto reduce el
    ancho de banda USB y permite alcanzar 12fps a 4000×3000. Sin MJPEG,
    a esa resolución el USB se satura y la tasa de frames cae drásticamente.
    Para activarlo en OpenCV usamos cv2.CAP_PROP_FOURCC con el código 'MJPG'.
"""

import os
import sys
import glob
import logging
import time
import numpy as np
import cv2


# Ruta a fotos_prueba/ — está en la raíz del proyecto (un nivel arriba de hardware/).
_RUTA_FOTOS_PRUEBA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fotos_prueba",
)


class Camara:
    """
    Encapsula la cámara ELP USB 48MP para el sistema de inspección.

    Uso básico:
        cam = Camara(config_camara)
        cam.abrir()
        frame = cam.capturar()  # devuelve numpy array (BGR)
        cam.cerrar()

    O usando el contexto de Python (más seguro, cierra sola):
        with Camara(config_camara) as cam:
            frame = cam.capturar()
    """

    def __init__(self, config: dict):
        """
        Constructor. Recibe la sección 'camara' del archivo calibracion.json.

        Parámetros:
            config (dict): Configuración de cámara. Ejemplo:
                {
                    "device_id": 1,           ← índice de dispositivo USB
                    "resolucion": [4000, 3000], ← ancho × alto en píxeles
                    "fps": 12                 ← fotogramas por segundo
                }
        """
        # Guardamos la configuración para usarla al abrir la cámara
        self._device_id  = config.get("device_id", 0)
        self._ancho      = config.get("resolucion", [4000, 3000])[0]  # 4000 px
        self._alto       = config.get("resolucion", [4000, 3000])[1]  # 3000 px
        self._fps        = config.get("fps", 12)

        # El objeto de VideoCapture de OpenCV (None = cámara no abierta todavía)
        self._cap = None

        # True si la cámara está abierta y lista para capturar
        self._abierta = False

        # True si no hay cámara real y estamos generando imágenes sintéticas
        self._modo_simulacion = False

        # Cache de rutas a fotos_prueba/ y puntero circular — solo se usa en
        # modo simulación. Se pobla perezosamente en la primera captura.
        self._fotos_sim: list = []
        self._indice_foto_sim: int = 0

        # Bandera: True cuando el frame simulado ya viene pre-recortado
        # (las fotos en fotos_prueba/ se guardan post-recorte), para que
        # capturar() no aplique el recorte una segunda vez.
        self._sim_frame_ya_recortado: bool = False

        # ── Recorte de imagen ──
        # Define la zona útil que se devuelve al capturar.
        # Si w == 0 o h == 0, se devuelve la imagen completa sin recortar.
        recorte = config.get("recorte", {})
        self._recorte_x = recorte.get("x", 0)
        self._recorte_y = recorte.get("y", 0)
        self._recorte_w = recorte.get("w", 0)
        self._recorte_h = recorte.get("h", 0)

    # ── Métodos de ciclo de vida ──────────────────────────────────────────────

    def abrir(self):
        """
        Abre la conexión con la cámara y configura resolución, FPS y formato.

        Este método debe llamarse antes de capturar imágenes.

        Lanza:
            RuntimeError si no se puede abrir la cámara y tampoco
            hay modo simulación configurado.
        """
        if self._abierta:
            logging.warning("Camara.abrir() llamado cuando ya estaba abierta.")
            return

        logging.info(
            f"Camara: abriendo dispositivo {self._device_id} "
            f"({self._ancho}×{self._alto} @ {self._fps}fps MJPEG)..."
        )

        # Abrir el dispositivo de video con el backend adecuado para cada plataforma.
        # En Linux (Jetson): CAP_V4L2 — driver nativo para cámaras USB.
        # En Windows: CAP_DSHOW (DirectShow) — el más compatible con cámaras USB.
        if sys.platform == "win32":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2
        self._cap = cv2.VideoCapture(self._device_id, backend)

        if not self._cap.isOpened():
            logging.error(
                f"Camara: no se pudo abrir dispositivo {self._device_id}. "
                "Activando modo simulación."
            )
            self._modo_simulacion = True
            self._abierta = True
            return

        # ── Configurar formato MJPEG ──
        # Le decimos a la cámara que comprima cada frame como JPEG antes de enviarlo.
        # Sin esto, USB2 no puede manejar 4000×3000 a 12fps (sería ~340MB/s sin comprimir).
        codigo_mjpeg = cv2.VideoWriter_fourcc(*"MJPG")
        self._cap.set(cv2.CAP_PROP_FOURCC, codigo_mjpeg)

        # ── Configurar resolución ──
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._ancho)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._alto)

        # ── Configurar FPS ──
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # ── Reducir el buffer interno del driver ──
        # V4L2 (Linux) guarda 4 frames por defecto; DSHOW (Windows) varía.
        # Con buffer=1 pedimos que guarde solo el más reciente.
        # No todos los drivers respetan este parámetro, por eso además
        # vaciamos manualmente en capturar() como respaldo.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verificar que la cámara aceptó la configuración
        # (algunos drivers redondean o ignoran valores exactos)
        ancho_real  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto_real   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_real    = self._cap.get(cv2.CAP_PROP_FPS)

        logging.info(
            f"Camara: configuración aplicada → "
            f"{ancho_real}×{alto_real} @ {fps_real:.1f}fps"
        )

        # Verificar si la resolución fue aceptada correctamente
        if ancho_real != self._ancho or alto_real != self._alto:
            logging.warning(
                f"Camara: la resolución solicitada ({self._ancho}×{self._alto}) "
                f"no coincide con la obtenida ({ancho_real}×{alto_real}). "
                "Verificar compatibilidad del driver."
            )

        # ── Calentar la cámara ──
        # A 4000×3000 MJPEG por USB2 el driver V4L2 tarda ~1-2 s en entregar
        # el primer frame válido. Sin el sleep inicial cada read() bloquea hasta
        # el timeout de select() (~3 s) imprimiendo warnings de V4L2.
        logging.info("Camara: descartando frames iniciales (calentamiento)...")
        time.sleep(2.0)
        for _ in range(5):
            self._cap.grab()  # grab() descarta sin decodificar — más rápido y no genera warning

        self._abierta = True
        logging.info("Camara: lista para capturar.")

    def cerrar(self):
        """
        Cierra la conexión con la cámara y libera los recursos.
        Llamar siempre al finalizar el programa.
        """
        if not self._abierta:
            return

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

        self._abierta = False
        self._modo_simulacion = False
        logging.info("Camara: cerrada y recursos liberados.")

    # ── Método principal: capturar imagen ─────────────────────────────────────

    def capturar(self, aplicar_recorte=True):
        """
        Captura un frame de la cámara.

        Parámetros:
            aplicar_recorte (bool): Si True (valor por defecto), aplica el recorte
                configurado y devuelve solo el área útil.
                Si False, devuelve la imagen completa sin recortar.
                Usar False cuando se quiere ver la imagen entera para definir
                o ajustar el área de recorte.

        Retorna:
            numpy.ndarray de forma (alto, ancho, 3) en formato BGR (Blue-Green-Red).
            Si hay recorte activo y aplicar_recorte=True, las dimensiones
            corresponden al área recortada, no a la resolución total de la cámara.

        Lanza:
            RuntimeError si la cámara no está abierta.
            RuntimeError si la captura falla 3 veces seguidas.
        """
        if not self._abierta:
            raise RuntimeError(
                "Camara.capturar(): la cámara no está abierta. "
                "Llamar a abrir() primero."
            )

        # Si estamos en modo simulación, devolver una imagen sintética
        if self._modo_simulacion:
            frame = self._generar_frame_simulado()
        else:
            # ── Vaciar el buffer antes de capturar ──
            # grab() descarta frames acumulados sin decodificarlos (más rápido
            # que read()). Así read() siempre devuelve el frame más reciente,
            # no uno de hace varios segundos. 4 pasadas cubren el buffer máximo
            # de V4L2; en Windows DSHOW el buffer suele ser menor, pero no hace daño.
            for _ in range(4):
                self._cap.grab()

            # Ahora leer el frame más reciente (buffer ya fue vaciado)
            frame = None
            for intento in range(3):
                ok, f = self._cap.read()
                if ok and f is not None:
                    frame = f
                    break
                logging.warning(
                    f"Camara.capturar(): intento {intento + 1}/3 falló. "
                    "Reintentando..."
                )
                time.sleep(0.05)  # 50ms de espera

            if frame is None:
                raise RuntimeError(
                    "Camara.capturar(): no se pudo obtener frame después de 3 intentos. "
                    "Verificar conexión USB."
                )

        # ── Corrección de distorsión del lente ──
        # Aplicamos cv2.remap con los mapas precalculados de la calibración
        # con tablero (calibracion_camara/calibracion.npz) SOBRE EL FRAME
        # COMPLETO.  Tiene que ser antes del recorte porque los coeficientes
        # de distorsión asumen (cx, cy) en el centro óptico del sensor
        # entero — aplicarlos sobre un frame recortado produciría una
        # corrección geométricamente incorrecta.
        #
        # El modo simulación no pasa por aquí (las fotos de fotos_prueba/
        # son de otra geometría y el frame sintético es de 400×300 — no
        # coincide con el tamaño de calibración, así que corregir_distorsion
        # los devuelve sin modificar de todas formas).
        from logica.cv_util import corregir_distorsion  # import tardío para evitar ciclos
        frame = corregir_distorsion(frame)

        # ── Aplicar recorte ──
        # Si el usuario configuró un área de inspección, recortamos el frame.
        # Esto reduce el tamaño de la imagen que procesa el Inspector,
        # hace las coordenadas de las ROIs más intuitivas y evita que
        # las zonas fuera de la etiqueta interfieran en el análisis.
        #
        # Excepción: si el frame viene de fotos_prueba/ (modo simulación) ya
        # está recortado a su área útil — recortar otra vez produciría una ROI
        # desplazada o minúscula.
        if (aplicar_recorte
                and self._recorte_w > 0 and self._recorte_h > 0
                and not self._sim_frame_ya_recortado):
            alto, ancho = frame.shape[:2]

            # Asegurar que las coordenadas no excedan los límites de la imagen.
            # Esto protege contra configuraciones inválidas o modo simulación
            # (cuya imagen es 400×300, mucho más pequeña que la real).
            x = max(0, min(self._recorte_x, ancho - 1))
            y = max(0, min(self._recorte_y, alto - 1))
            w = min(self._recorte_w, ancho - x)
            h = min(self._recorte_h, alto - y)

            if w > 0 and h > 0:
                frame = frame[y : y + h, x : x + w]

        return frame

    # ── Control del recorte ────────────────────────────────────────────────────

    def establecer_recorte(self, x: int, y: int, w: int, h: int):
        """
        Cambia el recorte activo sin necesidad de cerrar y reabrir la cámara.

        Se usa desde la GUI cuando el técnico define una nueva área de
        inspección en la pantalla de Mantenimiento.

        Parámetros:
            x, y (int): Esquina superior izquierda del recorte en la imagen
                        completa de la cámara (coordenadas en píxeles).
            w, h (int): Ancho y alto del recorte en píxeles.
                        Usar 0 para ambos si se quiere desactivar el recorte.
        """
        self._recorte_x = x
        self._recorte_y = y
        self._recorte_w = w
        self._recorte_h = h
        logging.info(
            f"Camara: recorte actualizado → "
            f"x={x} y={y} w={w} h={h}"
        )

    def quitar_recorte(self):
        """Desactiva el recorte. La próxima captura devolverá la imagen completa."""
        self.establecer_recorte(0, 0, 0, 0)
        logging.info("Camara: recorte eliminado. Se usará la imagen completa.")

    # ── Propiedades de solo lectura ────────────────────────────────────────────

    @property
    def esta_abierta(self):
        """True si la cámara está abierta y lista para capturar."""
        return self._abierta

    @property
    def en_simulacion(self):
        """True si la cámara está operando en modo simulación."""
        return self._modo_simulacion

    @property
    def resolucion(self):
        """Retorna una tupla (ancho, alto) con la resolución configurada."""
        return (self._ancho, self._alto)

    @property
    def tiene_recorte(self) -> bool:
        """True si hay un recorte activo configurado."""
        return self._recorte_w > 0 and self._recorte_h > 0

    @property
    def recorte(self) -> dict:
        """Retorna el recorte activo como dict {"x", "y", "w", "h"}."""
        return {
            "x": self._recorte_x, "y": self._recorte_y,
            "w": self._recorte_w, "h": self._recorte_h,
        }

    # ── Protocolo de contexto (with) ───────────────────────────────────────────
    # Permite usar: "with Camara(config) as cam:"
    # Al salir del bloque with (incluso con error), se llama a cerrar().

    def __enter__(self):
        """Se llama al entrar al bloque 'with'."""
        self.abrir()
        return self

    def __exit__(self, tipo_exc, valor_exc, traceback):
        """Se llama al salir del bloque 'with'. Siempre cierra la cámara."""
        self.cerrar()
        # Retornar False = si hubo una excepción, no la suprimimos.
        return False

    # ── Métodos privados ───────────────────────────────────────────────────────

    def _generar_frame_simulado(self):
        """
        Devuelve un frame para modo simulación.

        Prioridad:
            1. Si hay imágenes en fotos_prueba/, rota entre ellas (round-robin).
               Así una sesión de pruebas sin cámara puede iterar los casos reales.
            2. Si la carpeta está vacía o no existe, cae a un frame sintético
               (azul oscuro con texto) para que la app no crashee.
        """
        # Poblar cache de fotos la primera vez (lazy).
        if not self._fotos_sim and os.path.isdir(_RUTA_FOTOS_PRUEBA):
            self._fotos_sim = sorted(
                glob.glob(os.path.join(_RUTA_FOTOS_PRUEBA, "*.jpg"))
                + glob.glob(os.path.join(_RUTA_FOTOS_PRUEBA, "*.png"))
            )
            if self._fotos_sim:
                logging.info(
                    f"Camara: modo simulación usará {len(self._fotos_sim)} "
                    f"imágenes de fotos_prueba/ (rotación)."
                )

        # Rotar por fotos_prueba/ si hay disponibles.
        if self._fotos_sim:
            ruta = self._fotos_sim[self._indice_foto_sim % len(self._fotos_sim)]
            self._indice_foto_sim += 1
            frame = cv2.imread(ruta)
            if frame is not None:
                logging.info(
                    f"Camara (sim): {os.path.basename(ruta)} "
                    f"({frame.shape[1]}×{frame.shape[0]})"
                )
                # Las fotos en fotos_prueba/ ya están recortadas al área útil,
                # así que capturar() no debe aplicar su propio recorte.
                self._sim_frame_ya_recortado = True
                return frame
            logging.warning(f"Camara (sim): no pude leer '{ruta}'.")

        # Fallback sintético — solo si no hay fotos disponibles.
        self._sim_frame_ya_recortado = False
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (40, 40, 60)
        cv2.putText(frame, "CAMARA SIMULADA", (60, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (150, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1, cv2.LINE_AA)
        return frame
