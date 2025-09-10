# robot/sim_arm.py
from __future__ import annotations
import time
import threading
from typing import Dict, Any, Optional

from .base import RobotArm, Capabilities, RobotError


class SimArm(RobotArm):
    """
    Driver simulado para pruebas 'dry-run'.
    - Simula movimientos con tiempos proporcionales a la distancia.
    - Aplica límites XYZ y timeouts configurables.
    - Simula apertura/cierre de pinza.
    - No requiere hardware.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._cfg = config or {}
        # --- robot section ---
        robot_info = (self._cfg.get("robot") or {})
        self._name = robot_info.get("name", "SimArm v1")
        caps = robot_info.get("capabilities") or {}
        self._caps = Capabilities(
            named_poses=bool(caps.get("named_poses", True)),
            cartesian_moves=bool(caps.get("cartesian_moves", True)),
            gripper=(caps.get("gripper") or "servo_pwm"),
        )

        # --- motion / limits / gripper ---
        motion = (self._cfg.get("motion") or {})
        self._default_speed_mm_s: float = float(motion.get("default_speed_mm_s", 120.0))
        self._settle_ms: int = int(motion.get("settle_ms", 150))
        self._timeout_move_ms: int = int(motion.get("timeout_move_ms", 8000))

        limits = (self._cfg.get("limits") or {})
        self._x_lim = tuple(limits.get("x_mm", [0.0, 320.0]))
        self._y_lim = tuple(limits.get("y_mm", [0.0, 240.0]))
        self._z_lim = tuple(limits.get("z_mm", [0.0, 180.0]))

        grip = (self._cfg.get("gripper") or {})
        self._grip_open = float(grip.get("open", 10))
        self._grip_close = float(grip.get("close", 170))
        self._grip_action_ms = int(grip.get("action_ms", 300))

        # --- estado interno ---
        self._connected: bool = False
        self._state: str = "idle"       # idle | moving | error
        self._fault: Optional[str] = None
        self._last_pose_name: Optional[str] = None
        # Posición actual simulada
        self._pos = {"X": (self._x_lim[1] - self._x_lim[0]) / 2.0,
                     "Y": (self._y_lim[1] - self._y_lim[0]) / 2.0,
                     "Z": (self._z_lim[1] - self._z_lim[0]) / 2.0}
        self._gripper_state: str = "open"   # open | closed
        self._lock = threading.Lock()

    # --------- Información y capacidades ---------
    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    # --------- Ciclo de vida / conexión ----------
    def connect(self, **kwargs: Any) -> None:
        with self._lock:
            self._connected = True
            self._state = "idle"
            self._fault = None
        print("[SimArm] Conectado (simulado)")

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self._state = "idle"
        print("[SimArm] Desconectado (simulado)")

    def is_connected(self) -> bool:
        return self._connected

    # --------- Helpers internos ------------------
    def _sleep_ms(self, ms: int) -> None:
        time.sleep(max(ms, 0) / 1000.0)

    def _enforce_limits(self, x: float, y: float, z: float) -> None:
        if not (self._x_lim[0] <= x <= self._x_lim[1]):
            print(f"[SimArm] ERROR: X fuera de límites: {x} mm (perm: {self._x_lim})")
            raise RobotError(f"X fuera de límites: {x} mm (perm: {self._x_lim})")
        if not (self._y_lim[0] <= y <= self._y_lim[1]):
            print(f"[SimArm] ERROR: Y fuera de límites: {y} mm (perm: {self._y_lim})")
            raise RobotError(f"Y fuera de límites: {y} mm (perm: {self._y_lim})")
        if not (self._z_lim[0] <= z <= self._z_lim[1]):
            print(f"[SimArm] ERROR: Z fuera de límites: {z} mm (perm: {self._z_lim})")
            raise RobotError(f"Z fuera de límites: {z} mm (perm: {self._z_lim})")

    def _distance3(self, x: float, y: float, z: float) -> float:
        dx = x - self._pos["X"]
        dy = y - self._pos["Y"]
        dz = z - self._pos["Z"]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _move_to(self, x: float, y: float, z: float, speed_mm_s: float) -> bool:
        with self._lock:
            if not self._connected:
                print("[SimArm] ERROR: No conectado")
                raise RobotError("SimArm no conectado")
            self._enforce_limits(x, y, z)
            self._state = "moving"
            self._fault = None

        print(f"[SimArm] Moviendo a X={x:.1f} Y={y:.1f} Z={z:.1f} a {speed_mm_s:.1f} mm/s")
        dist = self._distance3(x, y, z)
        speed = max(speed_mm_s, 1e-3)
        duration_s = dist / speed

        # Simular movimiento con timeout
        if duration_s * 1000.0 > self._timeout_move_ms:
            print(f"[SimArm] ERROR: Timeout de movimiento ({duration_s*1000:.0f} ms > {self._timeout_move_ms} ms)")
            self._sleep_ms(self._timeout_move_ms)
            with self._lock:
                self._state = "error"
                self._fault = "timeout_move"
            return False

        # dormir el tiempo de movimiento y asentamiento
        self._sleep_ms(int(duration_s * 1000.0))
        self._sleep_ms(self._settle_ms)

        with self._lock:
            self._pos["X"], self._pos["Y"], self._pos["Z"] = x, y, z
            self._state = "idle"
            self._fault = None
        print(f"[SimArm] Llegó a destino X={x:.1f} Y={y:.1f} Z={z:.1f}")
        return True

    # --------- Poses y velocidades ---------------
    def home(self) -> bool:
        # Si hay pose HOME, usarla; si no, subir Z a un valor seguro y volver al centro.
        print("[SimArm] Moviendo a HOME")
        if "HOME" in self._poses:
            return self.move_named("HOME")
        center_x = (self._x_lim[0] + self._x_lim[1]) / 2.0
        center_y = (self._y_lim[0] + self._y_lim[1]) / 2.0
        safe_z = min(self._z_lim[1], max(self._z_lim[0], self._pos["Z"] + 50))
        return self._move_to(center_x, center_y, safe_z, self._default_speed_mm_s)

    def move_named(self, pose: str) -> bool:
        print(f"[SimArm] Moviendo a pose nombrada: {pose}")
        if not self._caps.named_poses:
            print(f"[SimArm] ERROR: Driver no soporta poses nombradas")
            raise RobotError("Este driver no soporta poses nombradas")
        target = self._poses.get(pose)
        if not target:
            print(f"[SimArm] ERROR: Pose desconocida: {pose}")
            raise RobotError(f"Pose desconocida: {pose}")

        x = float(target.get("X", self._pos["X"]))
        y = float(target.get("Y", self._pos["Y"]))
        z = float(target.get("Z", self._pos["Z"]))
        # F en mm/min → convertir a mm/s si existe; si no, usar default
        F = target.get("F")
        speed_mm_s = self._default_speed_mm_s if F is None else (float(F) / 60.0)

        ok = self._move_to(x, y, z, speed_mm_s)
        if ok:
            with self._lock:
                self._last_pose_name = pose
        return ok

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> bool:
        print(f"[SimArm] Movimiento cartesiano a X={x:.1f} Y={y:.1f} Z={z:.1f}")
        if not self._caps.cartesian_moves:
            print(f"[SimArm] ERROR: Driver no soporta movimientos cartesianos")
            raise RobotError("Este driver no soporta movimientos cartesianos")
        # Si viene 'speed', asumimos mm/min (consistente con G-code 'F')
        speed_mm_s = self._default_speed_mm_s if speed is None else (float(speed) / 60.0)
        return self._move_to(float(x), float(y), float(z), speed_mm_s)

    # --------- Pinza / efector final -------------
    def open_gripper(self) -> bool:
        if self._caps.gripper == "none":
            print("[SimArm] Pinza: no disponible (none)")
            return True
        print("[SimArm] Abriendo pinza...")
        self._sleep_ms(self._grip_action_ms)
        with self._lock:
            self._gripper_state = "open"
        print("[SimArm] Pinza abierta")
        return True

    def close_gripper(self) -> bool:
        if self._caps.gripper == "none":
            print("[SimArm] Pinza: no disponible (none)")
            return True
        print("[SimArm] Cerrando pinza...")
        self._sleep_ms(self._grip_action_ms)
        with self._lock:
            self._gripper_state = "closed"
        print("[SimArm] Pinza cerrada")
        return True

    # --------- Seguridad / estado ----------------
    def stop(self) -> None:
        print("[SimArm] Parada de emergencia (stop)")
        with self._lock:
            # En simulación: simplemente marcar estado y fault
            self._state = "idle"
            self._fault = "stopped"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "fault": self._fault,
                "last_pose": self._last_pose_name,
                "pos": dict(self._pos),
                "gripper": self._gripper_state,
                "name": self._name,
            }
