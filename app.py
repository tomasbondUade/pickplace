# app.py
from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print("Falta PyYAML. Instalá con: pip install pyyaml")
    raise

from robot.registry import create_robot_from_config
from planner.fsm import (
    PickPlaceFSM,
    VisionBundle,
    DispatchConfig,
    RunSafety,
    TelemetryOptions,
)


# ------------------------ Helpers de configuración ------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    print(f"[CONFIG] Cargando YAML: {path}")
    if not os.path.exists(path):
        print(f"[CONFIG] Archivo no encontrado: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML inválido (no es dict): {path}")
    print(f"[CONFIG] YAML cargado: {path}")
    return data


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[CONFIG] Actualizando configuración profunda...")
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config() -> Dict[str, Any]:
    print("[CONFIG] Iniciando carga de configuración...")
    base = _load_yaml("configs/default.yaml")

    # Overrides opcionales y específicos (si existen)
    vis = _load_yaml("configs/vision.yaml")
    disp = _load_yaml("configs/dispatch.yaml")

    # Merge simple: vision/dispatch pisan default en sus secciones
    cfg = dict(base)
    if vis:
        print("[CONFIG] Mezclando configuración de visión...")
        if "vision" not in cfg:
            cfg["vision"] = {}
        _deep_update(cfg["vision"], {
            "camera_id": vis.get("camera", {}).get("camera_id"),
            "resolution": vis.get("camera", {}).get("resolution"),
            "exposure_fixed": vis.get("camera", {}).get("exposure_fixed"),
            "white_balance_fixed": vis.get("camera", {}).get("white_balance_fixed"),
            "roi_px": vis.get("roi", {}).get("px"),
            "normalize_roi_with_fiducials": vis.get("roi", {}).get("normalize_with_fiducials"),
            "fiducials": vis.get("fiducials", {}).get("enabled"),
            "model_path": vis.get("classifier", {}).get("model_path"),
            "input_size": vis.get("classifier", {}).get("input_size"),
            "presence_method": vis.get("presence", {}).get("method"),
            "presence_stable_ms": vis.get("presence", {}).get("stable_ms"),
        })
    if disp:
        print("[CONFIG] Mezclando configuración de dispatch...")
        # Creamos una sección 'dispatch' compatible con DispatchConfig
        cfg["dispatch"] = {
            "classes": (disp.get("labels", {}).get("known") or cfg.get("dispatch", {}).get("classes") or []),
            "normalize": disp.get("labels", {}).get("normalize") or {},
            "map": disp.get("routing", {}).get("map") or {},
            "confidence_threshold": disp.get("policy", {}).get("confidence_threshold", cfg.get("dispatch", {}).get("confidence_threshold", 0.85)),
            "vote_frames": disp.get("policy", {}).get("vote_frames", cfg.get("dispatch", {}).get("vote_frames", 5)),
            "on_unknown": disp.get("policy", {}).get("on_unknown", "ROUTE"),
            "max_retries": disp.get("policy", {}).get("max_retries", 1),
            "retry_delay_ms": disp.get("policy", {}).get("retry_delay_ms", 400),
            "on_macro_fail": disp.get("safety", {}).get("on_macro_fail", "STOP"),
            "macro_retry": disp.get("safety", {}).get("macro_retry", 0),
        }
    print("[CONFIG] Configuración cargada correctamente.")
    return cfg


# ------------------------ Visión dummy (simulada) -------------------------
def make_dummy_vision(cfg: Dict[str, Any]) -> VisionBundle:
    """
    Visión simulada para probar end-to-end sin cámara ni modelo.
    - wait_presence: espera stable_ms y devuelve True (como si hubiera objeto).
    - capture_batch: devuelve una lista de 'frames' simbólicos.
    - classify_batch: siempre 'A' con 0.97 de confianza.
    """

    def wait_presence(stable_ms: int) -> bool:
        print(f"[VISION] Esperando presencia por {stable_ms} ms...")
        time.sleep(max(stable_ms, 0) / 1000.0)
        print("[VISION] Presencia detectada (simulada)")
        return True  # simular objeto presente

    def capture_batch(n_frames: int) -> List[Any]:
        print(f"[VISION] Capturando batch de {n_frames} frames (simulado)...")
        # Devolver 'n_frames' marcadores ficticios
        frames = [f"frame_{i}" for i in range(max(1, n_frames))]
        print(f"[VISION] Frames capturados: {frames}")
        return frames

    def classify_batch(frames: List[Any]) -> Tuple[str, float]:
        print(f"[VISION] Clasificando batch: {frames} (simulado)...")
        # Siempre clase A con alta confianza para validar PLACE(BOX_A)
        print("[VISION] Resultado: ('A', 0.97)")
        return "A", 0.97

    print("[VISION] Creando visión dummy (simulada)...")
    return VisionBundle(
        wait_presence=wait_presence,
        capture_batch=capture_batch,
        classify_batch=classify_batch,
    )


# ------------------------------ Main loop --------------------------------
def main() -> None:
    print("[APP] Iniciando main()...")
    cfg = load_config()

    print("[APP] Creando robot desde configuración...")
    arm = create_robot_from_config(cfg)

    print("[APP] Construyendo DispatchConfig...")
    dcfg_raw = cfg.get("dispatch") or {}
    dispatch = DispatchConfig(
        known_labels=list(dcfg_raw.get("classes") or []),
        normalize_map=dict(dcfg_raw.get("normalize") or {}),
        routing_map=dict(dcfg_raw.get("map") or {}),
        confidence_threshold=float(dcfg_raw.get("confidence_threshold", 0.85)),
        vote_frames=int(dcfg_raw.get("vote_frames", 5)),
        on_unknown=str(dcfg_raw.get("on_unknown", "ROUTE")),
        max_retries=int(dcfg_raw.get("max_retries", 1)),
        retry_delay_ms=int(dcfg_raw.get("retry_delay_ms", 400)),
        on_macro_fail=str(dcfg_raw.get("on_macro_fail", "STOP")),
        macro_retry=int(dcfg_raw.get("macro_retry", 0)),
    )

    print("[APP] Configurando seguridad operativa...")
    run_cfg = cfg.get("run", {})
    safety_cfg = (run_cfg.get("safety") or {})
    safety = RunSafety(
        z_safe=float(safety_cfg.get("z_safe", 120.0)),
        z_pick=float(safety_cfg.get("z_pick", 10.0)),
        z_drop=float(safety_cfg.get("z_drop", 20.0)),
    )

    print("[APP] Creando visión dummy...")
    vision = make_dummy_vision(cfg)

    print("[APP] Configurando telemetría...")
    telemetry = TelemetryOptions(
        log_images_enabled=bool((cfg.get("dispatch") or {}).get("log_images", True)),
        log_images_only_on_failure=True,
        save_dir=str((cfg.get("run") or {}).get("logging_dir", "runs/")),
        keep_last=500,
    )

    print("[APP] Inicializando FSM...")
    presence_stable_ms = int((cfg.get("vision") or {}).get("presence_stable_ms", 350))
    fsm = PickPlaceFSM(
        arm=arm,
        vision=vision,
        dispatch=dispatch,
        safety=safety,
        presence_stable_ms=presence_stable_ms,
        telemetry=telemetry,
        print_logs=True,
    )

    print("[APP] Iniciando loop principal. Ctrl+C para salir.")
    try:
        while True:
            print("[APP] Ejecutando paso de FSM...")
            state = fsm.step()
            print(f"[APP] Estado actual: {state}")
            if state == "STOP":
                print("[APP] Estado STOP detectado. Saliendo del loop.")
                break
            # Pequeño descanso para no saturar CPU
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[APP] Interrumpido por usuario. Enviando STOP...")
        fsm.stop()
    finally:
        try:
            print("[APP] Desconectando robot...")
            arm.disconnect()
        except Exception:
            print("[APP] Error al desconectar robot (ignorado)")
            pass
        print("[APP] Salida limpia.")


if __name__ == "__main__":
    main()
