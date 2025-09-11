
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time, yaml
from robot.registry import create_robot_from_config
from robot.base import RobotError

def load_cfg():
    with open("configs/default.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def clamp(v,a,b): return max(a, min(b, v))

def sweep_axis(arm, axis:str, step:float, direction:int, z_safe:float, soft_min:float, soft_max:float, max_moves:int=200):
    """Mueve en 'axis' en pasos 'step' (mm) con Z=z_safe. Devuelve último punto válido."""
    st = arm.status()
    pos = {k: float(v) for k,v in st.get("pos",{"X":0,"Y":0,"Z":0}).items()}
    # subir a z_safe
    arm.move_cartesian(pos.get("X",0), pos.get("Y",0), z_safe); time.sleep(0.1)
    moves = 0
    last_ok = pos.copy()
    while moves < max_moves:
        pos[axis] = clamp(pos.get(axis,0) + direction*step, soft_min, soft_max)
        ok = arm.move_cartesian(pos.get("X",0), pos.get("Y",0), z_safe)
        if not ok:
            print(f"[LIMIT] Movimiento falló en {axis}≈{pos[axis]:.2f}. Deteniendo.")
            break
        last_ok = pos.copy()
        moves += 1
        time.sleep(0.02)
    return last_ok

def main():
    cfg = load_cfg()
    arm = create_robot_from_config(cfg)
    z_safe = float(((cfg.get("run") or {}).get("safety") or {}).get("z_safe", 120))
    # Intentamos leer límites 'limits' del YAML del driver gcode (si existen)
    drv_path = (cfg.get("robot") or {}).get("config_file", "")
    limits = {"x_mm":[-9999,9999],"y_mm":[-9999,9999],"z_mm":[-9999,9999]}
    try:
        import yaml, os
        if drv_path and os.path.exists(drv_path):
            with open(drv_path,"r",encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            limits = (d.get("limits") or limits)
    except Exception:
        pass

    print("[PROBE] Conectando…")
    arm.connect()
    arm.home()

    # Empezar desde HOME a z_safe
    pos = arm.status().get("pos", {"X":0,"Y":0,"Z":0})
    arm.move_cartesian(pos.get("X",0), pos.get("Y",0), z_safe)

    step = 5.0  # mm: podés bajarlo a 2 mm si querés más fino
    print("[PROBE] Barrido +X…")
    last = sweep_axis(arm, "X", step, +1, z_safe, limits.get("x_mm",[ -9999,9999])[0], limits.get("x_mm",[ -9999,9999])[1])
    print("   último OK:", last)
    print("[PROBE] Barrido -X…")
    last = sweep_axis(arm, "X", step, -1, z_safe, limits.get("x_mm",[ -9999,9999])[0], limits.get("x_mm",[ -9999,9999])[1])
    print("   último OK:", last)

    print("[PROBE] Barrido +Y…")
    last = sweep_axis(arm, "Y", step, +1, z_safe, limits.get("y_mm",[ -9999,9999])[0], limits.get("y_mm",[ -9999,9999])[1])
    print("   último OK:", last)
    print("[PROBE] Barrido -Y…")
    last = sweep_axis(arm, "Y", step, -1, z_safe, limits.get("y_mm",[ -9999,9999])[0], limits.get("y_mm",[ -9999,9999])[1])
    print("   último OK:", last)

    print("[PROBE] Barrido +Z…")
    last = sweep_axis(arm, "Z", step, +1, z_safe, limits.get("z_mm",[ -9999,9999])[0], limits.get("z_mm",[ -9999,9999])[1])
    print("   último OK:", last)
    print("[PROBE] Barrido -Z… (cuidado: ajustá límite mínimo!)")
    last = sweep_axis(arm, "Z", step, -1, z_safe, limits.get("z_mm",[ 0,9999])[0], limits.get("z_mm",[ 0,9999])[1])
    print("   último OK:", last)

    print("[PROBE] Listo. Desconectando.")
    arm.disconnect()

if __name__ == "__main__":
    main()
