# robot/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any


GripperKind = Literal["servo_pwm", "vacuum", "relay", "sdk", "none"]


@dataclass(frozen=True)
class Capabilities:
    """
    Capacidades del brazo reportadas por el driver.
    - named_poses: soporta moverse a poses 'enseñadas' por nombre.
    - cartesian_moves: soporta movimientos cartesianos X/Y/Z (mm).
    - gripper: tipo de actuador de pinza disponible.
    """
    named_poses: bool
    cartesian_moves: bool
    gripper: GripperKind = "none"


class RobotError(RuntimeError):
    """Errores de driver/robot con mensaje claro para logs y UI."""
    pass


class RobotArm(ABC):
    """
    Interfaz común para todos los brazos.
    Las macros y la FSM solo dependen de estos métodos.
    """

    def __init__(self) -> None:
        self._poses: Dict[str, Dict[str, float]] = {}
        self._speeds: Dict[str, float] = {}

    # --------- Información y capacidades ---------
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre amigable del driver/robot (p.ej. 'SimArm v1')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Capacidades declaradas por el driver."""
        raise NotImplementedError

    # --------- Ciclo de vida / conexión ----------
    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Abre puertos/SDKs. Debe lanzar RobotError si falla."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Cierra recursos de forma segura."""
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """True si el robot está listo para recibir comandos."""
        raise NotImplementedError

    # --------- Poses y velocidades ---------------
    def load_poses(self, poses: Dict[str, Dict[str, float]]) -> None:
        """
        Carga poses nombradas desde YAML (HOME, PICK_SAFE, ...).
        Formato esperado: { 'POSE': { 'X': mm, 'Y': mm, 'Z': mm, 'F': mm/min } }
        """
        self._poses = poses or {}

    def set_speeds(self, speeds: Dict[str, float]) -> None:
        """
        Carga presets de velocidad (opcional).
        Ej.: { 'default': 1200, 'approach': 600 }
        """
        self._speeds = speeds or {}

    # --------- Comandos de movimiento ------------
    @abstractmethod
    def home(self) -> bool:
        """
        Lleva el robot a HOME si el hardware lo soporta.
        Retorna True si la acción se ejecutó correctamente.
        """
        raise NotImplementedError

    @abstractmethod
    def move_named(self, pose: str) -> bool:
        """
        Mueve a una pose nombrada cargada con load_poses().
        Debe validar que la pose exista y lanzar RobotError si no.
        """
        raise NotImplementedError

    @abstractmethod
    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Movimiento cartesiano (mm) opcionalmente con yaw y velocidad.
        Si el driver no soporta cartesiano, debe lanzar RobotError.
        """
        raise NotImplementedError

    # --------- Pinza / efector final -------------
    @abstractmethod
    def open_gripper(self) -> bool:
        """Abre la pinza (o desactiva vacío)."""
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self) -> bool:
        """Cierra la pinza (o activa vacío)."""
        raise NotImplementedError

    # --------- Seguridad / estado ----------------
    @abstractmethod
    def stop(self) -> None:
        """Parada segura inmediata (software). No debe colgarse."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """
        Devuelve estado mínimo para UI/logs, ej.:
        {
          "state": "idle|moving|error",
          "fault": None | "overcurrent" | "limit" | "...",
          "last_pose": "PICK_SAFE",
        }
        """
        raise NotImplementedError
