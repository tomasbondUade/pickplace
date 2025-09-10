# vision/camera.py
from __future__ import annotations
import time
import threading
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError("Falta OpenCV. Instalá con: pip install opencv-python") from e


class CameraError(RuntimeError):
    pass


class Camera:
    """
    Wrapper simple para OpenCV.
    - Abre la cámara con la resolución pedida.
    - Intenta fijar exposición y balance de blancos si se especifican.
    - Provee read() y close() thread-safe.
    - Tiene helper para extraer la ROI.
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        exposure_fixed: Optional[float] = None,      # ej: -6 (según driver)
        white_balance_fixed: Optional[float] = None  # ej: 4600 (Kelvin aprox)
    ) -> None:
        self.camera_id = int(camera_id)
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.exposure_fixed = exposure_fixed
        self.white_balance_fixed = white_balance_fixed

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._is_open = False

    # ---------------------- ciclo de vida ----------------------
    def open(self) -> None:
        # En Windows suele ser más estable CAP_DSHOW; si falla, fallback.
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)
        if not cap or not cap.isOpened():
            raise CameraError(f"No se pudo abrir la cámara id={self.camera_id}")

        # Intentar setear resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Desactivar autoexposición si podemos y fijar valor
        # Nota: según driver, CAP_PROP_AUTO_EXPOSURE usa escalas raras (0.25/0.75)
        try:
            if self.exposure_fixed is not None:
                # Apagar auto-exposure (valor depende del backend)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25=manual (en DSHOW)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure_fixed))
            else:
                # Dejar auto
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass  # no todos los drivers lo soportan

        # Balance de blancos
        try:
            if self.white_balance_fixed is not None:
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(self.white_balance_fixed))
            else:
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        except Exception:
            pass

        # Warmup: tirar algunos frames para estabilizar exposición/gain
        for _ in range(5):
            cap.read()
            time.sleep(0.03)

        with self._lock:
            self._cap = cap
            self._is_open = True

    def is_open(self) -> bool:
        with self._lock:
            return bool(self._is_open and self._cap is not None and self._cap.isOpened())

    def read(self) -> np.ndarray:
        with self._lock:
            if not self._cap or not self._is_open:
                raise CameraError("Cámara no abierta")
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise CameraError("Fallo al leer frame de la cámara")
            # frame es BGR (OpenCV)
            return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None
            self._is_open = False

    # ---------------------- utilitarios ----------------------
    @staticmethod
    def extract_roi(frame: np.ndarray, roi_px: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Recorta la ROI del frame (x, y, w, h) y la devuelve.
        Si la ROI cae en parte fuera del frame, se recorta a los límites válidos.
        """
        if frame is None or frame.size == 0:
            raise CameraError("Frame vacío en extract_roi")

        H, W = frame.shape[:2]
        x, y, w, h = map(int, roi_px)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            raise CameraError(f"ROI fuera de rango: {(x, y, w, h)} para frame {W}x{H}")
        return frame[y0:y1, x0:x1].copy()

    @staticmethod
    def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
