# scratch_cam_test.py
from __future__ import annotations
import os
import time
import yaml
import cv2
import numpy as np

from vision.camera import Camera, CameraError

def load_vision_cfg():
    path = "configs/vision.yaml"
    if not os.path.exists(path):
        # valores por defecto si todavía no tenés el YAML
        return {
            "camera": {"camera_id": 0, "resolution": [1280, 720], "exposure_fixed": None, "white_balance_fixed": None},
            "roi": {"px": [500, 200, 320, 320], "normalize_with_fiducials": False},
        }
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def main():
    cfg = load_vision_cfg()
    cam_cfg = cfg.get("camera", {})
    roi_cfg = cfg.get("roi", {})

    camera_id = cam_cfg.get("camera_id", 0)
    resolution = tuple(cam_cfg.get("resolution", [1280, 720]))
    exposure = cam_cfg.get("exposure_fixed", None)
    wb_temp = cam_cfg.get("white_balance_fixed", None)
    roi_px = tuple(roi_cfg.get("px", [500, 200, 320, 320]))

    cam = Camera(
        camera_id=camera_id,
        resolution=resolution,
        exposure_fixed=exposure,
        white_balance_fixed=wb_temp,
    )

    print(f"[TEST] Abriendo cámara id={camera_id} res={resolution} exp={exposure} wb={wb_temp}")
    cam.open()
    print("[TEST] Cámara abierta. q: salir | s: guardar frame+roi | ↑/↓/←/→: mover ROI | +/-: cambiar tamaño")

    # Parámetros editables en vivo para ajustar rápido la ROI
    x, y, w, h = map(int, roi_px)

    save_dir = "runs/camtest"
    os.makedirs(save_dir, exist_ok=True)
    n_saved = 0

    try:
        while True:
            frame = cam.read()  # BGR

            # Asegurar ROI dentro del frame
            H, W = frame.shape[:2]
            x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
            w = max(4, min(w, W - x)); h = max(4, min(h, H - y))

            # ROI recortada
            try:
                roi = Camera.extract_roi(frame, (x, y, w, h))
            except CameraError as e:
                roi = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(roi, "ROI fuera de rango", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Dibujar ROI sobre el frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Info de brillo (promedio)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(roi_gray))
            cv2.putText(frame, f"ROI=({x},{y},{w},{h})  Bright={mean_brightness:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 200, 20), 2)

            cv2.imshow("frame", frame)
            cv2.imshow("roi", roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # guardar frame y roi para inspección
                base = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(save_dir, f"{base}_frame.jpg"), frame)
                cv2.imwrite(os.path.join(save_dir, f"{base}_roi.jpg"), roi)
                n_saved += 1
                print(f"[TEST] Guardado #{n_saved} en {save_dir}")
            # Ajustes de ROI con teclado
            elif key == 82:   # ↑
                y = max(0, y - 5)
            elif key == 84:   # ↓
                y = min(H-1, y + 5)
            elif key == 81:   # ←
                x = max(0, x - 5)
            elif key == 83:   # →
                x = min(W-1, x + 5)
            elif key in (ord('+'), ord('=')):
                w = min(W - x, w + 5); h = min(H - y, h + 5)
            elif key == ord('-'):
                w = max(4, w - 5); h = max(4, h - 5)

    finally:
        cam.close()
        cv2.destroyAllWindows()
        print("[TEST] Cámara cerrada.")

if __name__ == "__main__":
    main()
