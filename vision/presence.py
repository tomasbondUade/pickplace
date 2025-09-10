# vision/presence.py
from __future__ import annotations
import time
from typing import Tuple, List, Any, Optional, Dict

import numpy as np
import cv2

from .camera import Camera, CameraError


class PresenceError(RuntimeError):
    pass


class PresenceDetector:
    """
    Detector de presencia en una ROI fija.

    Métodos:
      - init_background(): captura el fondo (solo para method='bgdiff')
      - wait_presence(stable_ms): bloquea hasta que haya presencia estable
      - capture_batch(n): devuelve n recortes ROI (BGR) para clasificar
      - measure(): calcula 'present' y métricas de la ROI (debug/telemetría)

    Parámetros clave:
      method: 'bgdiff' | 'edges'
      stable_ms: tiempo mínimo de presencia continua para aceptar (anti-ruido)
    """

    def __init__(
        self,
        camera: Camera,
        roi_px: Tuple[int, int, int, int],
        method: str = "bgdiff",
        stable_ms: int = 350,
        # --- bgdiff ---
        bg_frames: int = 10,             # frames para promedio/mediana de fondo
        bg_blur_ksize: int = 5,          # suavizado previo (impar)
        diff_thresh: int = 18,           # umbral de diferencia por píxel (0..255)
        min_changed_pixels: int = 1500,  # cantidad mínima de píxeles cambiados
        # --- edges ---
        edges_t1: int = 60,
        edges_t2: int = 140,
        edges_count_thresh: int = 400,   # cantidad mínima de píxeles "borde"
        # --- general ---
        cooldown_ms: int = 150,          # pausa breve entre chequeos
    ) -> None:
        self.cam = camera
        self.roi = tuple(map(int, roi_px))
        self.method = method.lower().strip()
        assert self.method in ("bgdiff", "edges"), "method debe ser 'bgdiff' o 'edges'"

        self.stable_ms = int(stable_ms)
        self.cooldown_ms = int(cooldown_ms)

        # Parámetros bgdiff
        self.bg_frames = int(bg_frames)
        self.bg_blur_ksize = int(bg_blur_ksize) if int(bg_blur_ksize) % 2 == 1 else int(bg_blur_ksize) + 1
        self.diff_thresh = int(diff_thresh)
        self.min_changed_pixels = int(min_changed_pixels)

        # Parámetros edges
        self.edges_t1 = int(edges_t1)
        self.edges_t2 = int(edges_t2)
        self.edges_count_thresh = int(edges_count_thresh)

        # Estado
        self._bg_gray: Optional[np.ndarray] = None  # solo para bgdiff

    # -------------------------- utilitarios --------------------------
    def _grab_roi_bgr(self) -> np.ndarray:
        frame = self.cam.read()  # BGR
        return Camera.extract_roi(frame, self.roi)

    @staticmethod
    def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------- inicialización ------------------------
    def init_background(self) -> None:
        """
        Captura el fondo para 'bgdiff' (promedia/mediana varios frames).
        Si el método es 'edges', no hace falta, pero no molesta llamarla.
        """
        if not self.cam.is_open():
            raise PresenceError("Cámara no abierta")

        rois = []
        for _ in range(max(1, self.bg_frames)):
            roi = self._grab_roi_bgr()
            gray = self._to_gray(roi)
            rois.append(gray)
            time.sleep(0.02)

        stack = np.stack(rois, axis=0)  # [N, H, W]
        # La mediana es más robusta frente a outliers que el promedio
        bg = np.median(stack, axis=0).astype(np.uint8)
        if self.bg_blur_ksize >= 3:
            bg = cv2.GaussianBlur(bg, (self.bg_blur_ksize, self.bg_blur_ksize), 0)
        self._bg_gray = bg

    # --------------------------- medición ----------------------------
    def measure(self) -> Dict[str, Any]:
        """
        Devuelve dict con:
          present: bool
          score: float (métrica principal)
          roi_size: (w, h)
          method: 'bgdiff' | 'edges'
        """
        roi = self._grab_roi_bgr()
        H, W = roi.shape[:2]

        if self.method == "bgdiff":
            if self._bg_gray is None:
                raise PresenceError("Fondo no inicializado (llamar a init_background())")
            gray = self._to_gray(roi)
            if self.bg_blur_ksize >= 3:
                gray = cv2.GaussianBlur(gray, (self.bg_blur_ksize, self.bg_blur_ksize), 0)

            # Diferencia absoluta y umbral binario
            diff = cv2.absdiff(gray, self._bg_gray)
            _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

            # Opcional: limpiar ruido con morfología ligera
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

            changed = int(cv2.countNonZero(mask))
            present = changed >= self.min_changed_pixels
            score = float(changed) / float(W * H)  # proporción de píxeles que cambiaron

            return {
                "present": bool(present),
                "score": float(score),
                "roi_size": (W, H),
                "method": "bgdiff",
                "changed_pixels": changed,
            }

        else:  # edges
            gray = self._to_gray(roi)
            edges = cv2.Canny(gray, self.edges_t1, self.edges_t2)
            count = int(cv2.countNonZero(edges))
            present = count >= self.edges_count_thresh
            score = float(count)

            return {
                "present": bool(present),
                "score": float(score),   # aquí el 'score' es el conteo de bordes
                "roi_size": (W, H),
                "method": "edges",
                "edges_count": count,
            }

    # ------------------- API para la FSM (VisionBundle) -------------------
    def wait_presence(self, stable_ms: Optional[int] = None) -> bool:
        """
        Bloquea hasta detectar presencia estable por 'stable_ms'.
        Devuelve True cuando la presencia se mantuvo el tiempo requerido.
        """
        req_ms = int(self.stable_ms if stable_ms is None else stable_ms)
        if self.method == "bgdiff" and self._bg_gray is None:
            # Capturar fondo si no existe
            self.init_background()

        present_since: Optional[float] = None

        while True:
            m = self.measure()
            if m["present"]:
                if present_since is None:
                    present_since = time.time()
                elapsed = (time.time() - present_since) * 1000.0
                if elapsed >= req_ms:
                    return True
            else:
                present_since = None

            time.sleep(max(self.cooldown_ms, 0) / 1000.0)

    def capture_batch(self, n_frames: int) -> List[Any]:
        """
        Captura 'n_frames' ROIs (BGR). Se usa luego para clasificar.
        """
        n = max(1, int(n_frames))
        out: List[np.ndarray] = []
        for _ in range(n):
            roi = self._grab_roi_bgr()
            out.append(roi)
            time.sleep(0.01)  # pequeña pausa para variar mínimamente
        return out
