from __future__ import annotations
import os, time, yaml
from robot.registry import create_robot_from_config
from robot.base import RobotError

def _load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    cfg = _load_yaml("configs/default.yaml")
    arm = create_robot_from_config(cfg)
    print("[TEST] Conectando…")
    arm.connect()
    print("[TEST]", arm.name, "conectado.")

    print("[TEST] HOME…")
    ok = arm.home()
    print("[TEST] HOME ok?", ok, arm.status())

    # Subir a z_safe y moverse un poco en X/Y
    z_safe = float(((cfg.get("run") or {}).get("safety") or {}).get("z_safe", 120))
    pos = arm.status().get("pos", {"X":0,"Y":0,"Z":0})
    ok = arm.move_cartesian(pos.get("X",0), pos.get("Y",0), z_safe)
    print("[TEST] Z->z_safe ok?", ok)

    print("[TEST] Jog +X 10 mm")
    ok = arm.move_cartesian(pos.get("X",0)+10, pos.get("Y",0), z_safe)
    print("  ->", ok)

    print("[TEST] Jog -X 10 mm (volver)")
    ok = arm.move_cartesian(pos.get("X",0), pos.get("Y",0), z_safe)
    print("  ->", ok)

    print("[TEST] Abrir/Cerrar pinza…")
    print(" open:", arm.open_gripper())
    time.sleep(0.3)
    print(" close:", arm.close_gripper())

    print("[TEST] Listo. Desconectando…")
    arm.disconnect()

if __name__ == "__main__":
    main()
