# macros/sequences.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

from robot.base import RobotArm, RobotError


@dataclass
class MacroResult:
    ok: bool
    error: Optional[str] = None


# ----------------- Helpers internos -----------------
def _sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def _move_named_or_raise(arm: RobotArm, pose: str) -> None:
    ok = arm.move_named(pose)
    if not ok:
        st = arm.status()
        fault = st.get("fault")
        raise RobotError(f"Fallo al mover a pose '{pose}' (fault={fault})")


# ----------------- Macros públicas ------------------
def HOME(arm: RobotArm) -> MacroResult:
    """
    Lleva el brazo a HOME. Si el driver no tiene HOME nativo,
    usa la pose 'HOME' cargada en YAML.
    """
    try:
        # Preferimos la pose HOME del YAML si existe
        try:
            _move_named_or_raise(arm, "HOME")
        except RobotError:
            # Fallback: usar home() del driver
            if not arm.home():
                st = arm.status()
                return MacroResult(False, f"HOME falló (fault={st.get('fault')})")
        return MacroResult(True)
    except RobotError as e:
        return MacroResult(False, str(e))


def PICK(
    arm: RobotArm,
    pick_safe: str = "PICK_SAFE",
    pick_z: str = "PICK_Z",
    open_before: bool = True,
    dwell_ms: int = 100,  # pequeña pausa para estabilizar
) -> MacroResult:
    """
    Rutina de pick estándar:
    1) Ir a PICK_SAFE
    2) (opcional) Abrir pinza
    3) Bajar a PICK_Z
    4) Cerrar pinza
    5) Subir a PICK_SAFE
    """
    try:
        _move_named_or_raise(arm, pick_safe)
        if open_before:
            if not arm.open_gripper():
                return MacroResult(False, "No se pudo abrir la pinza (PICK)")
            _sleep_ms(80)

        _move_named_or_raise(arm, pick_z)
        _sleep_ms(dwell_ms)

        if not arm.close_gripper():
            return MacroResult(False, "No se pudo cerrar la pinza (PICK)")

        _sleep_ms(80)
        _move_named_or_raise(arm, pick_safe)
        return MacroResult(True)
    except RobotError as e:
        return MacroResult(False, str(e))


def PLACE(
    arm: RobotArm,
    dest_base: str,          # Ej.: "BOX_A" | "BOX_B" | "BOX_C" | "BOX_REVIEW"
    dwell_ms: int = 100,
) -> MacroResult:
    """
    Rutina de place estándar hacia un destino base:
    Usa las poses '{dest_base}_SAFE' y '{dest_base}_Z'.
    """
    safe_pose = f"{dest_base}_SAFE"
    z_pose = f"{dest_base}_Z"

    try:
        _move_named_or_raise(arm, safe_pose)
        _move_named_or_raise(arm, z_pose)
        _sleep_ms(dwell_ms)

        if not arm.open_gripper():
            return MacroResult(False, f"No se pudo abrir la pinza (PLACE {dest_base})")

        _sleep_ms(80)
        _move_named_or_raise(arm, safe_pose)
        return MacroResult(True)
    except RobotError as e:
        return MacroResult(False, str(e))


def PICK_AND_PLACE(
    arm: RobotArm,
    dest_base: str,
    pick_safe: str = "PICK_SAFE",
    pick_z: str = "PICK_Z",
) -> MacroResult:
    """
    Macro combinada: PICK() + PLACE(dest_base).
    Útil para pruebas de extremo a extremo.
    """
    r1 = PICK(arm, pick_safe=pick_safe, pick_z=pick_z)
    if not r1.ok:
        return r1
    r2 = PLACE(arm, dest_base=dest_base)
    return r2


def STOP_ALL(arm: RobotArm) -> MacroResult:
    """
    Parada segura por software. No corta energía (eso debe ser un E-stop físico).
    """
    try:
        arm.stop()
        return MacroResult(True)
    except RobotError as e:
        return MacroResult(False, str(e))
