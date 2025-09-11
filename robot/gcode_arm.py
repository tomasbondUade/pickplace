# robot/gcode_arm.py
from __future__ import annotations
import time
import threading
from typing import Dict, Any, Optional

import serial  # pip install pyserial

from .base import RobotArm, Capabilities, RobotError


class GcodeArm(RobotArm):
    """
    Driver genérico G-code para brazos/firmwares tipo Marlin/GRBL/derivados.
    Configurable vía YAML (serial/protocol/motion/limits/gripper).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._cfg = config or {}
        pr = self._cfg.get("protocol") or {}
        self._axes = list(pr.get("axes", ["X","Y","Z"]))     # NUEVO
        self._move_tpl = pr.get("move_cmd_tpl", "G1 {COORDS} F{F:.1f}")  # ACTUALIZADO
        self._autodetect = bool((self._cfg.get("serial") or {}).get("autodetect", False))
        # --- robot info & caps ---
        rinfo = self._cfg.get("robot") or {}
        self._name = rinfo.get("name", "GCodeArm")
        caps = rinfo.get("capabilities") or {}
        self._caps = Capabilities(
            named_poses=bool(caps.get("named_poses", True)),
            cartesian_moves=bool(caps.get("cartesian_moves", True)),
            gripper=(caps.get("gripper") or "servo_pwm"),
        )

        # --- serial ---
        sc = self._cfg.get("serial") or {}
        self._port = sc.get("port") or ""
        self._baud = int(sc.get("baudrate", 115200))
        self._timeout = float(sc.get("timeout_s", 2.0))
        self._wtimeout = float(sc.get("write_timeout_s", 2.0))
        self._reset_on_connect = bool(sc.get("reset_on_connect", True))

        # --- protocol ---
        self._ok_token = (pr.get("ok_token") or "ok").lower()
        self._err_token = (pr.get("error_token") or "error").lower()
        self._nl = pr.get("newline", "\n")
        self._comment_prefix = pr.get("comment_prefix", ";")

        self._units_cmd = pr.get("units_cmd", "G21")
        self._abs_mode_cmd = pr.get("abs_mode_cmd", "G90")
        self._rel_mode_cmd = pr.get("rel_mode_cmd", "G91")
        self._home_cmd = pr.get("home_cmd", "G28")
        self._move_tpl = pr.get("move_cmd_tpl", "G1 X{X:.3f} Y{Y:.3f} Z{Z:.3f} F{F:.1f}")
        self._dwell_tpl = pr.get("dwell_cmd_tpl", "G4 P{ms}")
        self._stop_cmd = pr.get("stop_cmd", "M112")

        # --- motion/limits ---
        mo = self._cfg.get("motion") or {}
        self._default_f = float(mo.get("default_speed_mm_min", 1800))
        self._settle_ms = int(mo.get("settle_ms", 120))
        self._move_timeout_ms = int(mo.get("move_timeout_ms", 20000))

        lim = self._cfg.get("limits") or {}
        self._x_lim = tuple(lim.get("x_mm", [0.0, 320.0]))
        self._y_lim = tuple(lim.get("y_mm", [0.0, 240.0]))
        self._z_lim = tuple(lim.get("z_mm", [0.0, 180.0]))

        # --- gripper ---
        gr = self._cfg.get("gripper") or {}
        self._gripper_type = gr.get("type", "servo_pwm")
        self._open_pwm = float(gr.get("open_pwm", 10))
        self._close_pwm = float(gr.get("close_pwm", 170))
        self._grip_action_ms = int(gr.get("action_ms", 300))
        self._g_open_cmd = self._format_gripper_cmd(pr.get("gripper_open_cmd", "M3 S{open_pwm}"), self._open_pwm)
        self._g_close_cmd = self._format_gripper_cmd(pr.get("gripper_close_cmd", "M3 S{close_pwm}"), self._close_pwm)

        # --- estado ---
        self._ser: Optional[serial.Serial] = None
        self._connected = False
        self._state = "idle"
        self._fault: Optional[str] = None
        self._last_pose_name: Optional[str] = None
        self._pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}

        self._lock = threading.Lock()

        self._autodetect = bool((self._cfg.get("serial") or {}).get("autodetect", False))


    # ---------------- info/caps ----------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    # ---------------- conexión -----------------
    def connect(self, **kwargs: Any) -> None:
        # Autodetectar puerto si está vacío y fue habilitado en YAML (serial.autodetect: true)
        if (not self._port) and getattr(self, "_autodetect", False):
            try:
                import serial.tools.list_ports as lp
                candidates = []
                for p in lp.comports():
                    desc = (p.description or "").lower()
                    if any(k in desc for k in ("arduino", "ch340", "usb serial", "silabs", "cp210", "wch")):
                        candidates.append(p.device)
                if candidates:
                    self._port = candidates[0]
            except Exception as e:
                raise RobotError(f"No se pudo autodetectar un puerto serial: {e}")

        if not self._port:
            raise RobotError("Serial 'port' no configurado ni autodetectado (configs/robot_gcode.yaml)")

        # Abrir serial
        try:
            ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=self._timeout,
                write_timeout=self._wtimeout,
            )
        except Exception as e:
            raise RobotError(f"No se pudo abrir {self._port}: {e}") from e

        # Reset suave por DTR (si corresponde)
        if self._reset_on_connect:
            try:
                ser.setDTR(False); time.sleep(0.05); ser.setDTR(True)
            except Exception:
                pass

        # Tirar basura de arranque ~1s (muchos firmwares spamean al iniciar)
        t0 = time.time()
        try:
            while time.time() - t0 < 1.0:
                _ = ser.readline()
        except Exception:
            pass

        # Guardar handle y estado
        self._ser = ser
        self._connected = True
        self._state = "idle"
        self._fault = None

        # Flush y config inicial (mm + absoluto)
        self._flush_input()
        if self._units_cmd:
            self._send_and_wait_ok(self._units_cmd, timeout_ms=3000)
        if self._abs_mode_cmd:
            self._send_and_wait_ok(self._abs_mode_cmd, timeout_ms=3000)


    def disconnect(self) -> None:
        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None
            self._connected = False
            self._state = "idle"

    def is_connected(self) -> bool:
        return bool(self._connected and self._ser and self._ser.is_open)

    # ---------------- helpers internos ----------------
    def _format_gripper_cmd(self, tpl: str, pwm: float) -> str:
        return tpl.format(open_pwm=pwm, close_pwm=pwm)
    
    def _format_coords(self, target: dict) -> str:
        parts = []
        for ax in self._axes:
            if ax in target:
                val = float(target[ax])
                parts.append(f"{ax}{val:.3f}")
        return " ".join(parts)


    def _flush_input(self) -> None:
        if self._ser:
            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass

    def _send_line(self, line: str) -> None:
        if not self._ser or not self._ser.is_open:
            raise RobotError("Serial no abierto")
        data = (line + self._nl).encode("ascii", errors="ignore")
        try:
            self._ser.write(data)
            self._ser.flush()
        except Exception as e:
            raise RobotError(f"Error enviando G-code: {e}") from e

    def _readline(self) -> str:
        if not self._ser:
            return ""
        try:
            s = self._ser.readline().decode(errors="ignore").strip()
            return s
        except Exception:
            return ""

    def _wait_ok(self, timeout_ms: Optional[int] = None) -> bool:
        t0 = time.time()
        tout = (timeout_ms if timeout_ms is not None else self._move_timeout_ms) / 1000.0
        while (time.time() - t0) < tout:
            ln = self._readline().lower()
            if not ln:
                continue
            if self._ok_token in ln:
                return True
            if self._err_token and (self._err_token in ln):
                self._fault = ln
                return False
        self._fault = "timeout_move"
        return False

    def _send_and_wait_ok(self, cmd: str, timeout_ms: Optional[int] = None) -> bool:
        with self._lock:
            self._state = "moving"
            self._fault = None
        self._send_line(cmd)
        ok = self._wait_ok(timeout_ms)
        time.sleep(self._settle_ms / 1000.0)
        with self._lock:
            self._state = "idle" if ok else "error"
        return ok

    def _enforce_limits(self, x: float, y: float, z: float) -> None:
        if not (self._x_lim[0] <= x <= self._x_lim[1]):
            raise RobotError(f"X fuera de límites: {x} mm (perm: {self._x_lim})")
        if not (self._y_lim[0] <= y <= self._y_lim[1]):
            raise RobotError(f"Y fuera de límites: {y} mm (perm: {self._y_lim})")
        if not (self._z_lim[0] <= z <= self._z_lim[1]):
            raise RobotError(f"Z fuera de límites: {z} mm (perm: {self._z_lim})")

    # ---------------- movimientos ----------------
    def home(self) -> bool:
        if self._home_cmd:
            ok = self._send_and_wait_ok(self._home_cmd, timeout_ms=self._move_timeout_ms)
            return ok
        # Fallback: si no hay home, intentar ir a pose HOME si existe
        if "HOME" in self._poses:
            return self.move_named("HOME")
        return True  # sin home disponible

    def move_named(self, pose: str) -> bool:
        target = self._poses.get(pose)
        if not target:
            raise RobotError(f"Pose desconocida: {pose}")
        # límites: si hay X/E/Z/Y en el target, validalos (opcionalmente sólo X/Z)
        if "X" in target: self._enforce_limits(float(target["X"]), self._pos.get("Y",0.0), self._pos.get("Z",0.0))
        if "Z" in target: self._enforce_limits(self._pos.get("X",0.0), self._pos.get("Y",0.0), float(target["Z"]))

        F = float(target.get("F", self._default_f))
        coords = self._format_coords(target)
        cmd = self._move_tpl.format(COORDS=coords, F=F)
        ok = self._send_and_wait_ok(cmd)
        if ok:
            with self._lock:
                for ax in self._axes:
                    if ax in target:
                        self._pos[ax] = float(target[ax])
                self._last_pose_name = pose
        return ok

    def move_cartesian(
        self, x: float, y: float, z: float, yaw: Optional[float] = None, speed: Optional[float] = None
    ) -> bool:
        if not self._caps.cartesian_moves:
            raise RobotError("Este driver no soporta movimientos cartesianos")
        xf, yf, zf = float(x), float(y), float(z)
        self._enforce_limits(xf, yf, zf)
        F = float(speed) if speed is not None else self._default_f
        cmd = self._move_tpl.format(X=xf, Y=yf, Z=zf, F=F)
        ok = self._send_and_wait_ok(cmd)
        if ok:
            with self._lock:
                self._pos.update({"X": xf, "Y": yf, "Z": zf})
        return ok

    # ---------------- gripper -------------------
    def open_gripper(self) -> bool:
        if self._caps.gripper == "none":
            return True
        ok = self._send_and_wait_ok(self._g_open_cmd, timeout_ms=2000)
        time.sleep(self._grip_action_ms / 1000.0)
        return ok

    def close_gripper(self) -> bool:
        if self._caps.gripper == "none":
            return True
        ok = self._send_and_wait_ok(self._g_close_cmd, timeout_ms=2000)
        time.sleep(self._grip_action_ms / 1000.0)
        return ok

    # ---------------- seguridad/estado ---------
    def stop(self) -> None:
        if self._stop_cmd:
            try:
                self._send_line(self._stop_cmd)
            except Exception:
                pass
        with self._lock:
            self._state = "idle"
            self._fault = "stopped"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "fault": self._fault,
                "last_pose": self._last_pose_name,
                "pos": dict(self._pos),
                "name": self._name,
            }
