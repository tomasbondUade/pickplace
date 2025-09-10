# vision/classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os
import numpy as np
import cv2

# onnxruntime es opcional; si no está, usamos heurística
try:
    import onnxruntime as ort  # pip install onnxruntime
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False


@dataclass
class ClassifierConfig:
    model_path: Optional[str] = None         # ruta al .onnx (si hay)
    input_size: Tuple[int, int] = (224, 224) # (W, H)
    class_names: Tuple[str, ...] = ("A", "B", "C")
    # Normalización ImageNet por defecto
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Fallback heurístico (para mock o tests sin modelo)
    heuristic_map: Dict[str, str] = None     # p.ej. {"red":"A","blue":"B","green":"C"}


class BaseClassifier:
    def predict(self, frames_bgr: List[np.ndarray]) -> Tuple[str, float]:
        raise NotImplementedError


class HeuristicColorClassifier(BaseClassifier):
    """
    Clasificador simple por color dominante en HSV.
    Útil para probar sin modelo: mapea red/blue/green -> A/B/C.
    """
    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self.map = mapping or {"red": "A", "blue": "B", "green": "C"}

    @staticmethod
    def _color_label(bgr: np.ndarray) -> Tuple[str, float]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        # Máscara roja (dos rangos en HSV)
        m1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 80, 60), (179, 255, 255))
        red = (int(cv2.countNonZero(m1)) + int(cv2.countNonZero(m2)))
        # Azul
        blue = int(cv2.countNonZero(cv2.inRange(hsv, (90, 80, 60), (130, 255, 255))))
        # Verde
        green = int(cv2.countNonZero(cv2.inRange(hsv, (40, 80, 60), (80, 255, 255))))

        totals = {"red": red, "blue": blue, "green": green}
        label = max(totals, key=totals.get)
        total_px = bgr.shape[0] * bgr.shape[1]
        conf = float(totals[label]) / max(1.0, total_px)
        return label, conf

    def predict(self, frames_bgr: List[np.ndarray]) -> Tuple[str, float]:
        votes: Dict[str, float] = {}
        for img in frames_bgr:
            lab, conf = self._color_label(img)
            votes[lab] = votes.get(lab, 0.0) + conf
        label_raw = max(votes, key=votes.get)
        conf = float(votes[label_raw]) / max(1.0, sum(votes.values()))
        # mapear a clases del sistema si corresponde
        mapped = self.map.get(label_raw, "UNKNOWN")
        return mapped, conf


class ONNXClassifier(BaseClassifier):
    """
    Carga un modelo .onnx de clasificación (layout NCHW float32) y devuelve (label, conf).
    Asume softmax en salida; si no lo tiene, lo aplicamos nosotros.
    """
    def __init__(self, cfg: ClassifierConfig) -> None:
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime no disponible.")
        if not cfg.model_path or not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {cfg.model_path}")

        self.cfg = cfg
        providers = ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(cfg.model_path, providers=providers)

        meta = self._sess.get_inputs()[0]
        self._in_name = meta.name

        # salida: tomar primer output
        self._out_name = self._sess.get_outputs()[0].name

        self._mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(cfg.std,  dtype=np.float32).reshape(1, 1, 3)
        self._classes = list(cfg.class_names)

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        # BGR -> RGB, resize, [0,1], normalize, HWC->CHW
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.cfg.input_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        chw = np.transpose(img, (2, 0, 1))  # C,H,W
        return chw

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def predict(self, frames_bgr: List[np.ndarray]) -> Tuple[str, float]:
        if not frames_bgr:
            return "UNKNOWN", 0.0
        batch = np.stack([self._preprocess(f) for f in frames_bgr], axis=0)  # N,C,H,W
        ort_out = self._sess.run([self._out_name], {self._in_name: batch})[0]  # [N, C]
        probs = self._softmax(ort_out) if probs_needed(ort_out) else ort_out
        # Promediar probabilidades en N frames
        p = probs.mean(axis=0)  # [C]
        idx = int(np.argmax(p))
        conf = float(p[idx])
        if idx < 0 or idx >= len(self._classes):
            return "UNKNOWN", conf
        return self._classes[idx], conf


def probs_needed(arr: np.ndarray) -> bool:
    """Heurística: si alguna fila no suma ~1, aplicamos softmax."""
    row = arr[0]
    s = float(np.sum(row))
    return not (0.99 <= s <= 1.01)


def build_classifier(cfg: Dict[str, Any]) -> BaseClassifier:
    """
    Elige ONNX si hay modelo y onnxruntime; si no, usa heurística por color.
    Lee:
      - cfg['vision']['model_path'], cfg['vision']['input_size']
      - cfg['dispatch']['classes']
      - opcional: cfg['vision']['heuristic_map'] (red/blue/green -> clase)
    """
    vcfg = cfg.get("vision") or {}
    model_path = vcfg.get("model_path")
    input_size = tuple(vcfg.get("input_size", (224, 224)))

    classes = tuple((cfg.get("dispatch") or {}).get("classes") or ("A", "B", "C"))
    hmap = vcfg.get("heuristic_map") or {"red": "A", "blue": "B", "green": "C"}

    # Si hay modelo y ORT, usamos ONNX
    if _HAS_ORT and model_path and os.path.exists(model_path):
        return ONNXClassifier(ClassifierConfig(
            model_path=model_path,
            input_size=input_size,
            class_names=classes,
        ))
    # Fallback heurístico
    return HeuristicColorClassifier(mapping=hmap)
