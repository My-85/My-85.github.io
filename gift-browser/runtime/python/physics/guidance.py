"""A4 position-closed guidance over an immutable compiled H/P/R/V field.

The complete stream is still compiled without a rollout.  This module changes
only how that field is *indexed* during a real flight and adds a small
line-of-sight capture term in target space.  It never observes an actor,
checkpoint, teacher, optimizer, or action.

All positions use ``[north, east, altitude]`` metres.  Target rows use
``[heading, pitch, roll, airspeed]`` with angles in radians.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from numpy_compat import jnp, jax, at_set

from path_geometry import (
    PathGeometry,
    PhaseTracker,
)


A4_CLOSED_LOOP_GUIDANCE_SCHEMA = "planax.a4.position_closed_guidance.v3"
_EPS = 1.0e-9


def _wrap_pi(value: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class A4ClosedLoopGuidanceConfig:
    """One maneuver-independent A4 capture contract."""

    control_dt_s: float = 0.2
    geometry_spacing_m: float = 30.0
    tangent_smoothing_passes: int = 2
    projection_forward_window_m: float = 800.0
    preview_time_s: float = 2.4
    minimum_preview_m: float = 300.0
    maximum_preview_m: float = 800.0
    curvature_preview_gain: float = 1.6
    curvature_envelope_samples: int = 8
    maximum_heading_capture_deg: float = 15.0
    maximum_pitch_capture_deg: float = 10.0
    heading_slew_deg_s: float = 30.0
    pitch_slew_deg_s: float = 16.0
    roll_slew_deg_s: float = 50.0
    speed_slew_m_s2: float = 15.0
    completion_lead_time_s: float = 3.0
    completion_capture_radius_m: float = 300.0
    minimum_progress_fraction: float = 0.95

    def __post_init__(self) -> None:
        scalar_positive = (
            "control_dt_s",
            "geometry_spacing_m",
            "projection_forward_window_m",
            "preview_time_s",
            "minimum_preview_m",
            "maximum_preview_m",
            "heading_slew_deg_s",
            "pitch_slew_deg_s",
            "roll_slew_deg_s",
            "speed_slew_m_s2",
            "completion_lead_time_s",
            "completion_capture_radius_m",
        )
        for name in scalar_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "curvature_preview_gain",
            "maximum_heading_capture_deg",
            "maximum_pitch_capture_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.minimum_preview_m > self.maximum_preview_m:
            raise ValueError("minimum preview cannot exceed maximum preview")
        if (
            isinstance(self.tangent_smoothing_passes, bool)
            or not isinstance(self.tangent_smoothing_passes, int)
            or self.tangent_smoothing_passes < 0
        ):
            raise ValueError("tangent_smoothing_passes must be a nonnegative integer")
        if (
            isinstance(self.curvature_envelope_samples, bool)
            or not isinstance(self.curvature_envelope_samples, int)
            or self.curvature_envelope_samples < 2
        ):
            raise ValueError("curvature_envelope_samples must be an integer >=2")
        if self.maximum_heading_capture_deg > 45.0:
            raise ValueError("heading capture is outside the A4 local-capture domain")
        if self.maximum_pitch_capture_deg > 30.0:
            raise ValueError("pitch capture is outside the A4 local-capture domain")
        if (
            isinstance(self.minimum_progress_fraction, bool)
            or not math.isfinite(float(self.minimum_progress_fraction))
            or not 0.0 < float(self.minimum_progress_fraction) <= 1.0
        ):
            raise ValueError("minimum_progress_fraction must lie in (0,1]")

    @property
    def sha256(self) -> str:
        payload = {
            "schema": A4_CLOSED_LOOP_GUIDANCE_SCHEMA,
            "config": asdict(self),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class A4GuidanceCommand:
    """One auditable target and its geometry-only diagnostics."""

    target_hprv: np.ndarray
    raw_target_hprv: np.ndarray
    nominal_target_hprv: np.ndarray
    phase_s_m: float
    projected_phase_s_m: float
    phase_index: float
    terminal_phase_index: float
    projected_position_neu_m: np.ndarray
    cte_m: float
    lookahead_m: float
    curvature_envelope_per_m: float
    raw_heading_capture_rad: float
    raw_pitch_capture_rad: float
    applied_heading_capture_rad: float
    applied_pitch_capture_rad: float
    terminal_distance_m: float
    remaining_arc_m: float
    completion_just_triggered: bool
    geometrically_complete: bool
    slew_limited: np.ndarray


def validate_a4_guidance_inputs(
    reference_positions_neu_m: Any,
    reference_phase_m: Any,
    nominal_target_hprv: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immutable float64 copies after enforcing the coupled geometry ABI."""

    positions = np.asarray(reference_positions_neu_m)
    phase = np.asarray(reference_phase_m)
    targets = np.asarray(nominal_target_hprv)
    for name, value in (
        ("reference_positions_neu_m", positions),
        ("reference_phase_m", phase),
        ("nominal_target_hprv", targets),
    ):
        if not (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        ):
            raise TypeError(f"{name} must be real numeric")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 3:
        raise ValueError("reference positions must have shape [N>=3,3]")
    if phase.shape != (len(positions),):
        raise ValueError("reference phase must have shape [N]")
    if targets.shape != (len(positions), 4):
        raise ValueError("nominal target stream must have shape [N,4]")
    positions = np.asarray(positions, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (positions, phase, targets)):
        raise ValueError("guidance inputs must be finite")
    if not np.all(np.diff(phase) > 0.0) or abs(float(phase[0])) > 1.0e-6:
        raise ValueError("reference phase must start at zero and increase strictly")
    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(segment_length <= 1.0e-6):
        raise ValueError("reference geometry contains a zero-length segment")
    phase_delta = np.diff(phase)
    # The formal compiler stores arc length.  A relative tolerance covers the
    # final float32/float64 serialization boundary but rejects an arbitrary
    # time coordinate masquerading as distance.
    if not np.allclose(phase_delta, segment_length, rtol=2.0e-3, atol=2.0e-2):
        raise ValueError("reference phase is not the geometry arc length")
    if np.any(np.abs(targets[:, 0]) > math.pi + 1.0e-6):
        raise ValueError("nominal heading must use the [-pi,pi] representation")
    if np.any(np.abs(targets[:, 1]) > math.pi / 2.0 + 1.0e-6):
        raise ValueError("nominal pitch lies outside the physical representation")
    if np.any(np.abs(targets[:, 2]) > math.pi + 1.0e-6):
        raise ValueError("nominal roll must use the [-pi,pi] representation")
    if np.any(targets[:, 3] <= 0.0):
        raise ValueError("nominal speed must be positive")
    return positions.copy(), phase.copy(), targets.copy()


class A4ClosedLoopGuidance:
    """Stateful, monotone target compiler shared by teacher and student loops."""

    def __init__(
        self,
        reference_positions_neu_m: Any,
        reference_phase_m: Any,
        nominal_target_hprv: Any,
        config: A4ClosedLoopGuidanceConfig | None = None,
    ) -> None:
        positions, phase, targets = validate_a4_guidance_inputs(
            reference_positions_neu_m,
            reference_phase_m,
            nominal_target_hprv,
        )
        self.config = config or A4ClosedLoopGuidanceConfig()
        self.reference_positions_neu_m = positions
        self.reference_phase_m = phase
        self.nominal_target_hprv = targets
        self.geometry = PathGeometry(
            positions,
            spacing_m=self.config.geometry_spacing_m,
            tangent_smoothing_passes=self.config.tangent_smoothing_passes,
        )
        self.phase_tracker = PhaseTracker(
            self.geometry,
            control_dt_s=self.config.control_dt_s,
            forward_window_m=self.config.projection_forward_window_m,
        )
        self._geometry_to_reference_scale = (
            float(phase[-1]) / self.geometry.total_length_m
        )
        self._unwrapped_targets = targets.copy()
        self._unwrapped_targets[:, 0] = np.unwrap(targets[:, 0])
        self._unwrapped_targets[:, 2] = np.unwrap(targets[:, 2])
        self._previous_target: np.ndarray | None = None
        self._last_phase_s_m = 0.0
        self._completion_latched = False

    @property
    def geometry_sha256(self) -> str:
        material = "|".join(
            (
                A4_CLOSED_LOOP_GUIDANCE_SCHEMA,
                _array_sha256(self.reference_positions_neu_m),
                _array_sha256(self.reference_phase_m),
                _array_sha256(self.nominal_target_hprv),
            )
        )
        return hashlib.sha256(material.encode("ascii")).hexdigest()

    def reset(self, initial_target_hprv: Any) -> None:
        initial = np.asarray(initial_target_hprv, dtype=np.float64)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial target must be one finite H/P/R/V row")
        if initial[3] <= 0.0:
            raise ValueError("initial speed must be positive")
        self.phase_tracker.reset(0.0)
        self._previous_target = initial.copy()
        self._last_phase_s_m = 0.0
        self._completion_latched = False

    def _nominal_at(self, phase_s_m: float) -> np.ndarray:
        value = np.asarray(
            [
                np.interp(
                    phase_s_m,
                    self.reference_phase_m,
                    self._unwrapped_targets[:, channel],
                )
                for channel in range(4)
            ],
            dtype=np.float64,
        )
        value[0] = float(_wrap_pi(value[0]))
        value[2] = float(_wrap_pi(value[2]))
        return value

    def _curvature_adaptive_lookahead(
        self,
        geometry_phase_s_m: float,
        airspeed_mps: float,
    ) -> tuple[float, float]:
        config = self.config
        base = float(airspeed_mps) * config.preview_time_s
        stations = geometry_phase_s_m + np.linspace(
            0.0,
            min(base, self.geometry.total_length_m - geometry_phase_s_m),
            config.curvature_envelope_samples,
        )
        curvature = max(
            self.geometry.sample(float(station)).curvature_per_m
            for station in stations
        )
        adaptive = base / (1.0 + config.curvature_preview_gain * base * curvature)
        return (
            float(np.clip(adaptive, config.minimum_preview_m, config.maximum_preview_m)),
            float(curvature),
        )

    def _slew(self, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._previous_target is None:
            raise RuntimeError("guidance must be reset before its first step")
        previous = self._previous_target
        delta = desired - previous
        delta[0] = float(_wrap_pi(delta[0]))
        delta[2] = float(_wrap_pi(delta[2]))
        maximum = self.config.control_dt_s * np.asarray(
            [
                math.radians(self.config.heading_slew_deg_s),
                math.radians(self.config.pitch_slew_deg_s),
                math.radians(self.config.roll_slew_deg_s),
                self.config.speed_slew_m_s2,
            ],
            dtype=np.float64,
        )
        applied = np.clip(delta, -maximum, maximum)
        result = previous + applied
        result[0] = float(_wrap_pi(result[0]))
        result[2] = float(_wrap_pi(result[2]))
        limited = np.abs(delta) > maximum + 1.0e-12
        self._previous_target = result.copy()
        return result, limited

    def step(
        self,
        position_neu_m: Any,
        airspeed_mps: float,
    ) -> A4GuidanceCommand:
        """Compile one command from geometry, measured position, and speed."""

        if self._previous_target is None:
            raise RuntimeError("guidance must be reset before its first step")
        position = np.asarray(position_neu_m, dtype=np.float64)
        speed = float(airspeed_mps)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be one finite NEU vector")
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("airspeed must be finite and positive")

        geometry_phase, cte_m = self.phase_tracker.update(position, speed)
        if geometry_phase + 1.0e-9 < self._last_phase_s_m:
            raise RuntimeError("phase tracker violated monotonicity")
        self._last_phase_s_m = geometry_phase
        projected_reference_phase = float(
            np.clip(
                geometry_phase * self._geometry_to_reference_scale,
                0.0,
                self.reference_phase_m[-1],
            )
        )
        remaining_arc_m = max(
            0.0,
            float(self.reference_phase_m[-1] - projected_reference_phase),
        )
        completion_just_triggered = bool(
            not self._completion_latched
            and remaining_arc_m
            <= speed * self.config.completion_lead_time_s
            and cte_m <= self.config.completion_capture_radius_m
            and projected_reference_phase
            >= self.config.minimum_progress_fraction * self.reference_phase_m[-1]
        )
        self._completion_latched = bool(
            self._completion_latched or completion_just_triggered
        )
        command_phase = (
            float(self.reference_phase_m[-1])
            if self._completion_latched
            else projected_reference_phase
        )
        phase_index = float(
            np.interp(
                command_phase,
                self.reference_phase_m,
                np.arange(len(self.reference_phase_m), dtype=np.float64),
            )
        )
        projected = self.geometry.sample(geometry_phase).position_neu_m
        lookahead_m, curvature = self._curvature_adaptive_lookahead(
            geometry_phase, speed
        )
        carrot = self.geometry.sample(geometry_phase + lookahead_m).position_neu_m
        nominal_los = carrot - projected
        actual_los = carrot - position

        def los_angles(vector: np.ndarray, fallback: np.ndarray) -> tuple[float, float]:
            if float(np.linalg.norm(vector)) <= _EPS:
                vector = fallback
            horizontal = max(float(np.linalg.norm(vector[:2])), _EPS)
            return (
                math.atan2(float(vector[1]), float(vector[0])),
                math.atan2(float(vector[2]), horizontal),
            )

        tangent = self.geometry.sample(geometry_phase).tangent_neu
        nominal_heading, nominal_pitch = los_angles(nominal_los, tangent)
        actual_heading, actual_pitch = los_angles(actual_los, tangent)
        raw_heading_capture = float(_wrap_pi(actual_heading - nominal_heading))
        raw_pitch_capture = float(actual_pitch - nominal_pitch)
        bounded_heading_capture = float(
            np.clip(
                raw_heading_capture,
                -math.radians(self.config.maximum_heading_capture_deg),
                math.radians(self.config.maximum_heading_capture_deg),
            )
        )
        bounded_pitch_capture = float(
            np.clip(
                raw_pitch_capture,
                -math.radians(self.config.maximum_pitch_capture_deg),
                math.radians(self.config.maximum_pitch_capture_deg),
            )
        )

        # Completion is a one-way physical hand-off to the immutable terminal
        # row.  Continuing to apply a projected nonterminal row after the JAX
        # cursor has reached its endpoint makes the actor-visible target and
        # the environment termination target alternate every control cycle.
        # Once latched, both nominal and raw targets remain terminal; the
        # ordinary channel slew still provides a bounded transition.
        nominal = self._nominal_at(command_phase)
        raw = nominal.copy()
        heading_capture = (
            0.0 if self._completion_latched else bounded_heading_capture
        )
        pitch_capture = (
            0.0 if self._completion_latched else bounded_pitch_capture
        )
        raw[0] = float(_wrap_pi(raw[0] + heading_capture))
        raw[1] = float(raw[1] + pitch_capture)
        target, slew_limited = self._slew(raw.copy())
        terminal_distance = float(
            np.linalg.norm(position - self.reference_positions_neu_m[-1])
        )
        return A4GuidanceCommand(
            target_hprv=target,
            raw_target_hprv=raw,
            nominal_target_hprv=nominal,
            phase_s_m=command_phase,
            projected_phase_s_m=projected_reference_phase,
            phase_index=phase_index,
            terminal_phase_index=float(len(self.reference_phase_m) - 1),
            projected_position_neu_m=np.asarray(projected, dtype=np.float64),
            cte_m=float(cte_m),
            lookahead_m=lookahead_m,
            curvature_envelope_per_m=curvature,
            raw_heading_capture_rad=raw_heading_capture,
            raw_pitch_capture_rad=raw_pitch_capture,
            applied_heading_capture_rad=heading_capture,
            applied_pitch_capture_rad=pitch_capture,
            terminal_distance_m=terminal_distance,
            remaining_arc_m=remaining_arc_m,
            completion_just_triggered=completion_just_triggered,
            geometrically_complete=self._completion_latched,
            slew_limited=np.asarray(slew_limited, dtype=np.bool_),
        )


def bind_a4_guidance_to_stream_state(state: Any, command: A4GuidanceCommand, *, control_dt_s: float, previous_guidance_state: Any | None=None) -> Any:
    """Bind one host guidance result to an existing single-A4 stream state.

    This function changes only target/cursor bookkeeping.  Aircraft and
    actuator state are untouched.  Importing JAX lazily keeps the standalone
    guidance/diagnostic module device-neutral.
    """
    dt = float(control_dt_s)
    if not math.isfinite(dt) or dt <= np.float32(0.0):
        raise ValueError('control_dt_s must be finite and positive')
    target_host = np.asarray(command.target_hprv, dtype=np.float64)
    if target_host.shape != (4,) or not np.all(np.isfinite(target_host)):
        raise ValueError('guidance command target must be one finite H/P/R/V row')
    if target_host[3] <= np.float32(0.0):
        raise ValueError('guidance command target speed must be positive')
    if not math.isfinite(float(command.phase_index)) or not math.isfinite(float(command.terminal_phase_index)) or command.terminal_phase_index < np.float32(2.0) or (not np.float32(0.0) <= command.phase_index <= command.terminal_phase_index):
        raise ValueError('guidance command phase metadata is invalid')
    target = jnp.asarray(target_host[None, :], dtype=jnp.float32)
    if tuple(np.shape(state.stream_next_target)) != (1, 4):
        raise ValueError('closed-loop guidance supports exactly one A4 aircraft')
    previous_state = state if previous_guidance_state is None else previous_guidance_state
    if tuple(np.shape(previous_state.stream_next_target)) != (1, 4):
        raise ValueError('previous guidance state must contain exactly one target')
    previous = jnp.asarray(previous_state.stream_next_target, dtype=jnp.float32)
    delta = target - previous
    delta = at_set(delta, (slice(None, None, None), 0), jnp.arctan2(jnp.sin(delta[:, 0]), jnp.cos(delta[:, 0])))
    delta = at_set(delta, (slice(None, None, None), 2), jnp.arctan2(jnp.sin(delta[:, 2]), jnp.cos(delta[:, 2])))
    rates = delta / jnp.asarray(dt, dtype=jnp.float32)
    old_phase = jnp.asarray(previous_state.stream_cursor.phase, dtype=jnp.float32)
    phase_step = jnp.asarray(previous_state.stream_cursor.phase_step, dtype=jnp.float32)
    terminal_phase = jnp.asarray([command.terminal_phase_index], dtype=jnp.float32)
    open_phase_cap = jnp.maximum(np.float32(0.0), terminal_phase - phase_step - jnp.asarray(np.float32(0.001), dtype=jnp.float32))
    requested_phase = jnp.asarray([command.phase_index], dtype=jnp.float32)
    requested_phase = jnp.where(jnp.asarray(command.geometrically_complete, dtype=jnp.bool_), terminal_phase, jnp.minimum(requested_phase, open_phase_cap))
    new_phase = requested_phase
    new_phase = jnp.maximum(old_phase, new_phase)
    complete = jnp.asarray([command.geometrically_complete], dtype=jnp.bool_)
    completion_age = jnp.where(complete, previous_state.stream_completion_age + jnp.asarray(1, dtype=jnp.int32), jnp.zeros_like(previous_state.stream_completion_age))
    cursor = state.stream_cursor.replace(phase=new_phase, current_target=target)
    return state.replace(target_heading=target[:, 0], target_pitch=target[:, 1], target_roll=target[:, 2], target_vt=target[:, 3], stream_cursor=cursor, stream_current_target=target, stream_next_target=target, stream_target_rates=rates, stream_complete=complete, stream_completion_age=completion_age, stream_phase_advance=new_phase - old_phase)


__all__ = [
    "A4_CLOSED_LOOP_GUIDANCE_SCHEMA",
    "A4ClosedLoopGuidance",
    "A4ClosedLoopGuidanceConfig",
    "A4GuidanceCommand",
    "bind_a4_guidance_to_stream_state",
    "validate_a4_guidance_inputs",
]
