# robot/registry.py
from __future__ import annotations
from typing import Any, Dict
import os

try:
    import yaml  # PyYAML
except Exception as e:
    raise RuntimeError("Falta PyYAML. Instalá 'pyyaml'.") from e

from .base import RobotArm, RobotError
from .sim_arm import SimArm
from .gcode_arm import GcodeArm   # cuando lo implementes
# from .dobot_arm import DobotArm   # cuando lo implementes
# from .ur_arm import URArm         # cuando lo implementes
# from .xarm_arm import XArm        # cuando lo implementes


def _load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise RobotError(f"No se encontró el archivo YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RobotError(f"YAML inválido (no es dict): {path}")
    return data


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge recursivo simple: src pisa a dst en claves escalares y mezcla dicts."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


DRIVER_MAP = {
    "sim": SimArm,
    "gcode": GcodeArm,
    # "dobot": DobotArm,
    # "ur": URArm,
    # "xarm": XArm,
}


def create_robot_from_config(cfg: Dict[str, Any]) -> RobotArm:
    """
    Crea e inicializa el driver del robot a partir de la config ya cargada
    (p. ej., fusión de configs/default.yaml + overrides).
    - Lee robot.config_file (parámetros del driver).
    - Lee robot.poses_file (poses nombradas y speeds).
    - Instancia el driver indicado en robot.driver.
    """
    if not isinstance(cfg, dict):
        raise RobotError("Config raíz inválida (esperado dict).")

    robot_cfg = (cfg.get("robot") or {})
    driver_key = (robot_cfg.get("driver") or "").strip().lower()
    if not driver_key:
        raise RobotError("Falta 'robot.driver' en la configuración.")

    driver_cls = DRIVER_MAP.get(driver_key)
    if driver_cls is None:
        raise RobotError(f"Driver desconocido: '{driver_key}'. "
                         f"Drivers soportados: {', '.join(DRIVER_MAP.keys())}")

    # Cargar YAML específico del driver (parámetros)
    driver_yaml_path = robot_cfg.get("config_file")
    driver_yaml = _load_yaml(driver_yaml_path) if driver_yaml_path else {}

    # Instanciar el driver con su config
    arm: RobotArm = driver_cls(config=driver_yaml)

    # Cargar poses nombradas y speeds (si hay archivo)
    poses_yaml_path = robot_cfg.get("poses_file")
    if poses_yaml_path:
        poses_yaml = _load_yaml(poses_yaml_path)
        poses_block = poses_yaml.get("poses") or {}
        speeds_block = poses_yaml.get("speeds") or {}
        if not isinstance(poses_block, dict):
            raise RobotError(f"'poses' inválido en {poses_yaml_path}")
        if speeds_block and not isinstance(speeds_block, dict):
            raise RobotError(f"'speeds' inválido en {poses_yaml_path}")

        arm.load_poses(poses_block)
        arm.set_speeds(speeds_block)

    return arm
