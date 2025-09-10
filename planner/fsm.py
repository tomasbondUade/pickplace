# planner/fsm.py
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any, Literal

from robot.base import RobotArm, RobotError
from macros.sequences import HOME, PICK, PLACE, STOP_ALL, MacroResult


# ----------- Tipos de visión inyectables (no importan frameworks) -----------
# Esperamos que la capa de visión provea estas funciones simples.
WaitPresenceFn = Callable[[int], bool]                  # stable_ms -> bool
CaptureBatchFn = Callable[[int], List[Any]]             # n_frames -> frames
ClassifyBatchFn = Callable[[List[Any]], Tuple[str, float]]  # frames -> (label, conf)


@dataclass
class VisionBundle:
    """Conjunto de callables que expone la capa de visión a la FSM."""
    wait_presence: WaitPresenceFn
    capture_batch: CaptureBatchFn
    classify_batch: ClassifyBatchFn


@dataclass
class DispatchConfig:
    """Config de despacho (ver configs/dispatch.yaml)."""
    known_labels: List[str]
    normalize_map: Dict[str, str]
    routing_map: Dict[str, str]
    confidence_threshold: float
    vote_frames: int
    on_unknown: Literal["ROUTE", "RETRY", "SKIP"] = "ROUTE"
    max_retries: int = 1
    retry_delay_ms: int = 400
    on_macro_fail: Literal["STOP", "SKIP_ITEM", "RETRY_MACRO"] = "STOP"
    macro_retry: int = 0


@dataclass
class RunSafety:
    """Config de seguridad operativa (subset de configs/default.yaml:run.safety)."""
    z_safe: float = 120.0
    z_pick: float = 10.0
    z_drop: float = 20.0


@dataclass
class TelemetryOptions:
    log_images_enabled: bool = True
    log_images_only_on_failure: bool = True
    save_dir: str = "runs/telemetry/"
    keep_last: int = 500


# ----------------------------- Estados de la FSM -----------------------------
State = Literal["INIT", "IDLE", "CAPTURE", "CLASSIFY", "EXECUTE", "RETURN", "ERROR", "STOP"]


@dataclass
class FSMStatus:
    state: State
    last_label: Optional[str] = None
    last_conf: Optional[float] = None
    retries: int = 0
    last_error: Optional[str] = None


# ----------------------------- Máquina de estados ----------------------------
class PickPlaceFSM:
    """
    Orquesta el ciclo:
    INIT → IDLE → (presencia) → CAPTURE → CLASSIFY → EXECUTE → RETURN → IDLE
    Maneja UNKNOWN y fallas de macros según política de dispatch.
    """

    def __init__(
        self,
        arm: RobotArm,
        vision: VisionBundle,
        dispatch: DispatchConfig,
        safety: Optional[RunSafety] = None,
        presence_stable_ms: int = 350,
        telemetry: Optional[TelemetryOptions] = None,
        print_logs: bool = True,
    ) -> None:
        self.arm = arm
        self.vision = vision
        self.dispatch = dispatch
        self.safety = safety or RunSafety()
        self.presence_stable_ms = presence_stable_ms
        self.telemetry = telemetry or TelemetryOptions()
        self.print_logs = print_logs

        self._status = FSMStatus(state="INIT")
        self._stop_event = threading.Event()

    # ---------------------------- Utilitarios --------------------------------
    def _log(self, msg: str) -> None:
        if self.print_logs:
            print(f"[FSM] {msg}")

    def _norm_label(self, label: str) -> str:
        if label is None:
            return "UNKNOWN"
        s = str(label)
        n = self.dispatch.normalize_map.get(s, self.dispatch.normalize_map.get(s.lower(), s))
        # Si después de normalizar no está en known, lo tratamos como UNKNOWN
        return n if n in self.dispatch.known_labels or n == "UNKNOWN" else "UNKNOWN"

    def _route_for(self, label: str) -> str:
        return self.dispatch.routing_map.get(label, "BOX_REVIEW")

    # ---------------------------- Ciclo de vida --------------------------------
    def stop(self) -> None:
        self._stop_event.set()
        try:
            STOP_ALL(self.arm)
        except Exception:
            pass
        self._status.state = "STOP"

    def status(self) -> FSMStatus:
        return self._status

    # ------------------------------ Pasos -------------------------------------
    def step(self) -> State:
        """
        Ejecuta una transición de estado. Llamar en loop hasta que devuelva 'STOP'.
        """
        if self._stop_event.is_set():
            self._status.state = "STOP"
            return "STOP"

        st = self._status.state

        # ----- INIT -----
        if st == "INIT":
            try:
                if not self.arm.is_connected():
                    self.arm.connect()
                r = HOME(self.arm)
                if not r.ok:
                    raise RobotError(r.error or "HOME falló")
                self._status = FSMStatus(state="IDLE")
                self._log("INIT → IDLE")
            except RobotError as e:
                self._status = FSMStatus(state="ERROR", last_error=str(e))
                self._log(f"INIT error: {e}")
            return self._status.state

        # ----- IDLE -----
        if st == "IDLE":
            present = False
            try:
                present = self.vision.wait_presence(self.presence_stable_ms)
            except Exception as e:
                self._status = FSMStatus(state="ERROR", last_error=f"wait_presence: {e}")
                self._log(f"IDLE error: {e}")
                return "ERROR"

            if present:
                self._status.state = "CAPTURE"
                self._log("IDLE → CAPTURE")
            else:
                # quedarse en IDLE
                self._sleep_ms(50)
            return self._status.state

        # ----- CAPTURE -----
        if st == "CAPTURE":
            try:
                frames = self.vision.capture_batch(max(1, self.dispatch.vote_frames))
                # (Opcional) guardar frames si querés telemetría siempre
                self._cached_frames = frames
                self._status.state = "CLASSIFY"
                self._log("CAPTURE → CLASSIFY")
            except Exception as e:
                self._status = FSMStatus(state="ERROR", last_error=f"capture_batch: {e}")
                self._log(f"CAPTURE error: {e}")
            return self._status.state

        # ----- CLASSIFY -----
        if st == "CLASSIFY":
            try:
                frames = getattr(self, "_cached_frames", None)
                if not frames:
                    # Si no hay frames en cache por algún motivo, recapturamos
                    frames = self.vision.capture_batch(max(1, self.dispatch.vote_frames))
                    self._cached_frames = frames

                raw_label, conf = self.vision.classify_batch(frames)
                label = self._norm_label(raw_label)
                if conf < self.dispatch.confidence_threshold:
                    label = "UNKNOWN"

                self._status.last_label = label
                self._status.last_conf = float(conf)

                if label == "UNKNOWN":
                    # Política ante UNKNOWN
                    if self.dispatch.on_unknown == "RETRY" and self._status.retries < self.dispatch.max_retries:
                        self._status.retries += 1
                        self._log(f"CLASSIFY: UNKNOWN (conf={conf:.2f}) → RETRY {self._status.retries}/{self.dispatch.max_retries}")
                        self._sleep_ms(self.dispatch.retry_delay_ms)
                        self._status.state = "IDLE"  # volver a esperar presencia estable
                        return "IDLE"
                    elif self.dispatch.on_unknown == "SKIP":
                        self._log(f"CLASSIFY: UNKNOWN (conf={conf:.2f}) → SKIP item")
                        self._status.state = "RETURN"
                        return "RETURN"
                    else:
                        # ROUTE a BOX_REVIEW
                        self._log(f"CLASSIFY: UNKNOWN (conf={conf:.2f}) → ROUTE a BOX_REVIEW")
                        self._status.state = "EXECUTE"
                        return "EXECUTE"

                # Etiqueta válida
                self._log(f"CLASSIFY: label={label} conf={conf:.2f} → EXECUTE")
                self._status.state = "EXECUTE"
            except Exception as e:
                self._status = FSMStatus(state="ERROR", last_error=f"classify_batch: {e}")
                self._log(f"CLASSIFY error: {e}")
            return self._status.state

        # ----- EXECUTE -----
        if st == "EXECUTE":
            label = self._status.last_label or "UNKNOWN"
            dest_base = self._route_for(label)
            try:
                r1 = PICK(self.arm)
                if not r1.ok:
                    self._handle_macro_fail("PICK", r1)
                    return self._status.state

                r2 = PLACE(self.arm, dest_base=dest_base)
                if not r2.ok:
                    self._handle_macro_fail(f"PLACE({dest_base})", r2)
                    return self._status.state

                self._status.state = "RETURN"
                self._log(f"EXECUTE OK → RETURN")
            except RobotError as e:
                self._status = FSMStatus(state="ERROR", last_error=str(e))
                self._log(f"EXECUTE error: {e}")
            return self._status.state

        # ----- RETURN -----
        if st == "RETURN":
            r = HOME(self.arm)
            if not r.ok:
                self._status = FSMStatus(state="ERROR", last_error=r.error or "HOME falló")
                self._log(f"RETURN error: {self._status.last_error}")
                return "ERROR"
            # Reset de reintentos y vuelta a IDLE
            self._status.retries = 0
            self._status.state = "IDLE"
            self._log("RETURN → IDLE")
            return "IDLE"

        # ----- ERROR -----
        if st == "ERROR":
            err = self._status.last_error or "desconocido"
            self._log(f"ERROR: {err} → STOP")
            self.stop()
            return "STOP"

        # ----- STOP -----
        return "STOP"

    # --------------------------- Helpers internos -----------------------------
    def _handle_macro_fail(self, which: str, res: MacroResult) -> None:
        msg = res.error or "macro falló"
        self._log(f"{which} FAIL: {msg}")
        policy = self.dispatch.on_macro_fail
        if policy == "RETRY_MACRO" and self.dispatch.macro_retry > 0:
            # Reintento simple: una vez
            self._log(f"Reintentando {which} (1/{self.dispatch.macro_retry})")
            # NOTA: por simplicidad, no recursivizamos; dejamos a la app
            # manejar re-ejecución de step() y política superior si se necesita.
        elif policy == "SKIP_ITEM":
            self._status.state = "RETURN"
            self._status.last_error = f"{which} fail (skipped)"
        else:
            self._status = FSMStatus(state="ERROR", last_error=f"{which} fail: {msg}")

    @staticmethod
    def _sleep_ms(ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)
