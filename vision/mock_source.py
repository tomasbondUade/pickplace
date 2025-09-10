# vision/mock_source.py
from __future__ import annotations
import time
from typing import Optional, Tuple
import numpy as np
import cv2

class MockCamera:
    """
    Cámara simulada:
    - Produce un frame constante con ruido leve (fondo).
    - Cada 'period_s' segundos alterna entre ROI vacía y ROI con un "objeto" (círculo).
    - Interfaz compatible en lo básico con vision.camera.Camera: open(), is_open(), read(), close().
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        roi_px: Tuple[int, int, int, int] = (500, 200, 320, 320),
        period_s: float = 3.0,
        add_noise: bool = True,
        draw_roi_border: bool = False,
        exposure_fixed: Optional[float] = None,
        white_balance_fixed: Optional[float] = None,
    ) -> None:
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.roi = tuple(map(int, roi_px))
        self.period_s = float(period_s)
        self.add_noise = bool(add_noise)
        self.draw_roi_border = bool(draw_roi_border)
        self._open = False
        self._t0 = time.time()

    def open(self) -> None:
        self._open = True
        self._t0 = time.time()  # reiniciar ciclo

    def is_open(self) -> bool:
        return self._open

    def read(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError("MockCamera no está abierta")
        t = time.time() - self._t0
        present = int(t // self.period_s) % 2 == 1  # alterna cada period_s

        # Fondo: gris claro con ruido leve
        frame = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        if self.add_noise:
            noise = np.random.randint(0, 6, size=frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

        x, y, w, h = self.roi

        if self.draw_roi_border:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if present:
            # “Objeto” rojo en el centro de la ROI
            cx, cy = x + w // 2, y + h // 2
            radius = max(6, min(w, h) // 4)
            cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)

        return frame

    def close(self) -> None:
        self._open = False
