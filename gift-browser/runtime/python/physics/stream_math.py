from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Dict
import numpy as np
from numpy_compat import jnp, jax, at_set, ReplaceMixin
Array = np.ndarray

TARGET_DIM=4
FUTURE_TWIST_HORIZON_STEPS=4
FUTURE_TWIST_RATE_DEG_S=55.0
FUTURE_TWIST_CODE_SCALE=0.5
FUTURE_TWIST_CODE_MAX_NORM=0.50001

def wrap_pi(angle: Array) -> Array:
    """Wrap radians to ``[-pi, pi)`` without leaving JAX."""
    angle = jnp.asarray(angle)
    return jnp.mod(angle + jnp.pi, np.float32(2.0) * jnp.pi) - jnp.pi

def angular_difference(to_angle: Array, from_angle: Array) -> Array:
    """Shortest signed angular displacement ``from_angle -> to_angle``."""
    return wrap_pi(jnp.asarray(to_angle) - jnp.asarray(from_angle))

@dataclass(frozen=True)
class StreamBounds(ReplaceMixin):
    """Registered target-domain and per-second slew limits."""
    heading_slew_rad_s: float = np.deg2rad(np.float32(35.0))
    pitch_slew_rad_s: float = np.deg2rad(np.float32(20.0))
    roll_slew_rad_s: float = np.deg2rad(np.float32(60.0))
    speed_slew_m_s2: float = np.float32(20.0)
    pitch_limit_rad: float = np.deg2rad(np.float32(85.0))
    roll_limit_rad: float = np.pi
    min_speed_m_s: float = np.float32(140.0)
    max_speed_m_s: float = np.float32(340.0)

@dataclass(frozen=True)
class StreamCursor(ReplaceMixin):
    """Batched, augmentation-aware cursor into a padded target bank.

    All fields have a leading batch dimension.  ``phase`` is the one scalar
    coordinate shared by heading, pitch, roll, and speed.  ``phase_step`` is
    the number of source samples traversed in one policy control interval.
    """
    stream_id: Array
    phase: Array
    phase_step: Array
    mirror_sign: Array
    amplitude_scale: Array
    time_scale: Array
    heading_rotation: Array
    anchor_phase: Array
    anchor_target: Array
    current_target: Array
    certified_fidelity: Array

@dataclass(frozen=True)
class StreamTargetBatch(ReplaceMixin):
    """Current/next HPRV targets and their finite-difference rates."""
    current: Array
    next: Array
    rates: Array
    complete: Array
    phase_advance: Array

def unwrap_padded_target_bank(padded_targets: Array, lengths: Array) -> Array:
    """Unwrap valid heading/roll sequences while leaving padding irrelevant."""
    targets = jnp.asarray(padded_targets, dtype=jnp.float32)
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    max_points = targets.shape[1]

    def unwrap_channel(channel: Array) -> Array:
        first = channel[:, :1]
        raw_delta = angular_difference(channel[:, 1:], channel[:, :-1])
        valid_delta = jnp.arange(max_points - 1)[None, :] < lengths[:, None] - 1
        delta = jnp.where(valid_delta, raw_delta, jnp.zeros_like(raw_delta))
        return jnp.concatenate([first, first + jnp.cumsum(delta, axis=1)], axis=1)
    heading = unwrap_channel(targets[..., 0])
    roll = unwrap_channel(targets[..., 2])
    return at_set(at_set(targets, (..., 0), heading), (..., 2), roll)

def _gather_interpolated_rows(unwrapped_targets: Array, lengths: Array, stream_ids: Array, phases: Array) -> Array:
    """Gather one linearly interpolated row per batch element."""
    bank = jnp.asarray(unwrapped_targets, dtype=jnp.float32)
    lengths = jnp.asarray(lengths, dtype=jnp.int32)
    ids = jnp.asarray(stream_ids, dtype=jnp.int32)
    phase = jnp.asarray(phases, dtype=jnp.float32)
    chosen_lengths = lengths[ids]
    max_phase = jnp.maximum(chosen_lengths - 1, 0).astype(jnp.float32)
    phase = jnp.clip(phase, np.float32(0.0), max_phase)
    i0 = jnp.floor(phase).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, chosen_lengths - 1)
    batch_index = jnp.arange(ids.shape[0])
    row0 = bank[ids, i0]
    row1 = bank[ids, i1]
    weight = (phase - i0.astype(jnp.float32))[:, None]
    return row0 + weight * (row1 - row0)

def _transformed_raw_targets(cursor: StreamCursor, unwrapped_targets: Array, lengths: Array, phases: Array, bounds: StreamBounds) -> Array:
    """Apply rotation/mirror/amplitude transforms about the exact t0 anchor."""
    rows = _gather_interpolated_rows(unwrapped_targets, lengths, cursor.stream_id, phases)
    base_anchor = _gather_interpolated_rows(unwrapped_targets, lengths, cursor.stream_id, cursor.anchor_phase)
    amplitude = cursor.amplitude_scale
    mirror = cursor.mirror_sign
    anchor = cursor.anchor_target
    heading = wrap_pi(cursor.heading_rotation + mirror * amplitude * rows[:, 0])
    at_anchor = phases == cursor.anchor_phase
    heading = jnp.where(at_anchor, anchor[:, 0], heading)
    augmented_pitch = anchor[:, 1] + amplitude * (rows[:, 1] - base_anchor[:, 1])
    augmented_roll = wrap_pi(anchor[:, 2] + mirror * amplitude * (rows[:, 2] - base_anchor[:, 2]))
    augmented_speed = anchor[:, 3] + amplitude * (rows[:, 3] - base_anchor[:, 3])
    certified = cursor.certified_fidelity
    pitch = jnp.where(certified, rows[:, 1], augmented_pitch)
    roll = jnp.where(certified, wrap_pi(mirror * rows[:, 2]), augmented_roll)
    speed = jnp.where(certified, rows[:, 3], augmented_speed)
    pitch = jnp.clip(pitch, -bounds.pitch_limit_rad, bounds.pitch_limit_rad)
    roll = jnp.clip(roll, -bounds.roll_limit_rad, bounds.roll_limit_rad)
    speed = jnp.clip(speed, bounds.min_speed_m_s, bounds.max_speed_m_s)
    return jnp.stack([heading, pitch, roll, speed], axis=-1)

def target_rates(current_target: Array, next_target: Array, control_dt_s: float) -> Array:
    """Finite-difference H/P/R/V rates with periodic heading and roll."""
    current = jnp.asarray(current_target, dtype=jnp.float32)
    nxt = jnp.asarray(next_target, dtype=jnp.float32)
    dt = jnp.maximum(jnp.asarray(control_dt_s, dtype=jnp.float32), np.float32(1e-06))
    delta = nxt - current
    delta = at_set(delta, (..., 0), angular_difference(nxt[..., 0], current[..., 0]))
    delta = at_set(delta, (..., 2), angular_difference(nxt[..., 2], current[..., 2]))
    return jnp.nan_to_num(delta / dt, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0))

def _stream_target_batch_from_unwrapped(cursor: StreamCursor, unwrapped_targets: Array, lengths: Array, bounds: StreamBounds, control_dt_s: float) -> StreamTargetBatch:
    chosen_lengths = jnp.asarray(lengths, dtype=jnp.int32)[cursor.stream_id]
    last_phase = jnp.maximum(chosen_lengths - 1, 0).astype(jnp.float32)
    next_knot = jnp.floor(cursor.phase) + np.float32(1.0)
    proposed_phase = jnp.minimum(cursor.phase + cursor.phase_step, last_phase)
    allowed = jnp.maximum(jnp.asarray(control_dt_s, dtype=jnp.float32), np.float32(1e-06)) * jnp.asarray([bounds.heading_slew_rad_s, bounds.pitch_slew_rad_s, bounds.roll_slew_rad_s, bounds.speed_slew_m_s2], dtype=jnp.float32)

    def phase_path_variation(phase: Array) -> Array:
        first_end_phase = jnp.minimum(phase, next_knot)
        first_end = _transformed_raw_targets(cursor, unwrapped_targets, lengths, first_end_phase, bounds)
        candidate = _transformed_raw_targets(cursor, unwrapped_targets, lengths, phase, bounds)
        first_delta = first_end - cursor.current_target
        first_delta = at_set(first_delta, (..., 0), angular_difference(first_end[..., 0], cursor.current_target[..., 0]))
        first_delta = at_set(first_delta, (..., 2), angular_difference(first_end[..., 2], cursor.current_target[..., 2]))
        second_delta = candidate - first_end
        second_delta = at_set(second_delta, (..., 0), angular_difference(candidate[..., 0], first_end[..., 0]))
        second_delta = at_set(second_delta, (..., 2), angular_difference(candidate[..., 2], first_end[..., 2]))
        crossed_knot = phase > next_knot
        return jnp.abs(first_delta) + jnp.where(crossed_knot[..., None], jnp.abs(second_delta), np.float32(0.0))

    def phase_is_slew_feasible(phase: Array) -> Array:
        return jnp.all(phase_path_variation(phase) <= allowed + np.float32(1e-07), axis=-1)
    full_feasible = phase_is_slew_feasible(proposed_phase)

    def bisect_step(_: int, interval: tuple[Array, Array]) -> tuple[Array, Array]:
        (lower, upper) = interval
        middle = np.float32(0.5) * (lower + upper)
        feasible = phase_is_slew_feasible(middle)
        return (jnp.where(feasible, middle, lower), jnp.where(feasible, upper, middle))
    if np.all(full_feasible):
        (admitted_phase, _) = (proposed_phase, proposed_phase)
    else:
        (admitted_phase, _) = jax.lax.fori_loop(0, 24, bisect_step, (cursor.phase, proposed_phase))
    advanced_phase = jnp.where(full_feasible, proposed_phase, admitted_phase)
    phase_advance = advanced_phase - cursor.phase
    bounded_next = _transformed_raw_targets(cursor, unwrapped_targets, lengths, advanced_phase, bounds)
    rates = target_rates(cursor.current_target, bounded_next, control_dt_s)
    final_raw = _transformed_raw_targets(cursor, unwrapped_targets, lengths, last_phase, bounds)
    final_delta = final_raw - cursor.current_target
    final_delta = at_set(final_delta, (..., 0), angular_difference(final_raw[..., 0], cursor.current_target[..., 0]))
    final_delta = at_set(final_delta, (..., 2), angular_difference(final_raw[..., 2], cursor.current_target[..., 2]))
    at_last_phase = cursor.phase >= last_phase - np.float32(1e-06)
    settled = jnp.max(jnp.abs(final_delta), axis=-1) <= np.float32(1e-05)
    return StreamTargetBatch(current=cursor.current_target, next=bounded_next, rates=rates, complete=at_last_phase & settled, phase_advance=phase_advance)

def stream_target_batch(cursor: StreamCursor, padded_targets: Array, lengths: Array, bounds: StreamBounds, control_dt_s: float) -> StreamTargetBatch:
    """Public pure-JAX current/next/rate evaluation for a cursor."""
    unwrapped = unwrap_padded_target_bank(padded_targets, lengths)
    return _stream_target_batch_from_unwrapped(cursor, unwrapped, lengths, bounds, control_dt_s)

def advance_stream_cursor(cursor: StreamCursor, padded_targets: Array, lengths: Array, bounds: StreamBounds, control_dt_s: float) -> Tuple[StreamCursor, StreamTargetBatch]:
    """Advance every batch element once, preserving its coupled cursor."""
    unwrapped = unwrap_padded_target_bank(padded_targets, lengths)
    before = _stream_target_batch_from_unwrapped(cursor, unwrapped, lengths, bounds, control_dt_s)
    chosen_lengths = jnp.asarray(lengths, dtype=jnp.int32)[cursor.stream_id]
    last_phase = jnp.maximum(chosen_lengths - 1, 0).astype(jnp.float32)
    next_phase = jnp.minimum(cursor.phase + before.phase_advance, last_phase)
    next_cursor = cursor.replace(phase=next_phase, current_target=before.next)
    after = _stream_target_batch_from_unwrapped(next_cursor, unwrapped, lengths, bounds, control_dt_s)
    return (next_cursor, after)

def _hpr_to_quaternion_nb(hpr: Array) -> Array:
    """Convert [..., heading, pitch, roll] to body-to-NED quaternions."""
    values = jnp.asarray(hpr, dtype=jnp.float32)
    (heading, pitch, roll) = (values[..., 0], values[..., 1], values[..., 2])
    (cy, sy) = (jnp.cos(np.float32(0.5) * heading), jnp.sin(np.float32(0.5) * heading))
    (cp, sp) = (jnp.cos(np.float32(0.5) * pitch), jnp.sin(np.float32(0.5) * pitch))
    (cr, sr) = (jnp.cos(np.float32(0.5) * roll), jnp.sin(np.float32(0.5) * roll))
    return jnp.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)

def _quaternion_conjugate(quaternion: Array) -> Array:
    value = jnp.asarray(quaternion)
    return value * jnp.asarray([np.float32(1.0), -np.float32(1.0), -np.float32(1.0), -np.float32(1.0)], dtype=value.dtype)

def _quaternion_multiply(left: Array, right: Array) -> Array:
    (lw, lx, ly, lz) = jnp.moveaxis(jnp.asarray(left), -1, 0)
    (rw, rx, ry, rz) = jnp.moveaxis(jnp.asarray(right), -1, 0)
    return jnp.stack([lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw], axis=-1)

def _relative_so3_log(start_q_nb: Array, end_q_nb: Array) -> Array:
    """Return the shortest ``start -> end`` SO(3) log in the start frame.

    Both inputs are body-to-NED quaternions.  ``conjugate(start) * end`` is
    therefore expressed in the starting target body frame, which is exactly
    the coordinate frame registered for the future-twist observation.  The
    nonnegative-scalar hemisphere makes the returned rotation the shortest
    one and removes the quaternion double-cover sign ambiguity.
    """
    start = jnp.asarray(start_q_nb, dtype=jnp.float32)
    end = jnp.asarray(end_q_nb, dtype=jnp.float32)
    start = start / jnp.maximum(jnp.linalg.norm(start, axis=-1, keepdims=True), np.float32(1e-08))
    end = end / jnp.maximum(jnp.linalg.norm(end, axis=-1, keepdims=True), np.float32(1e-08))
    relative = _quaternion_multiply(_quaternion_conjugate(start), end)
    relative = relative / jnp.maximum(jnp.linalg.norm(relative, axis=-1, keepdims=True), np.float32(1e-08))
    relative = jnp.where(relative[..., :1] < np.float32(0.0), -relative, relative)
    scalar = jnp.clip(relative[..., 0], np.float32(0.0), np.float32(1.0))
    vector = relative[..., 1:]
    vector_norm = jnp.linalg.norm(vector, axis=-1)
    angle = np.float32(2.0) * jnp.arctan2(vector_norm, scalar)
    safe_norm = jnp.where(vector_norm > np.float32(1e-07), vector_norm, np.float32(1.0))
    multiplier = jnp.where(vector_norm > np.float32(1e-07), angle / safe_norm, np.float32(2.0))
    return vector * multiplier[..., None]

def future_twist_preview_semantic_validity(preview_code: Array, maximum_norm: float=FUTURE_TWIST_CODE_MAX_NORM) -> Array:
    """Return one explicit semantic-validity bit per preview row.

    Invalid values are reported, not repaired.  In particular this helper
    never clips or replaces a nonfinite or over-bound preview, so a trainer or
    deployment wrapper can fail the whole rollout closed.
    """
    code = jnp.asarray(preview_code, dtype=jnp.float32)
    if code.ndim < 1 or code.shape[-1] != 3:
        raise ValueError('future-twist preview code must have shape [..., 3]')
    bound = jnp.asarray(maximum_norm, dtype=jnp.float32)
    return jnp.all(jnp.isfinite(code), axis=-1) & jnp.isfinite(bound) & (bound >= np.float32(0.0)) & (jnp.linalg.norm(code, axis=-1) <= bound)

def future_twist_preview_code(cursor: StreamCursor, padded_targets: Array, lengths: Array, bounds: StreamBounds, control_dt_s: float, *, horizon_steps: int=FUTURE_TWIST_HORIZON_STEPS, rate_limit_deg_s: float=FUTURE_TWIST_RATE_DEG_S, code_scale: float=FUTURE_TWIST_CODE_SCALE, zero_ablation: bool=False) -> Tuple[Array, Array]:
    """Compile the causal ``T[k+1] -> T[k+1+horizon]`` SO(3) preview.

    The endpoint is obtained by invoking :func:`advance_stream_cursor` for
    every control, rather than indexing raw bank phase.  This is essential for
    fractional/augmented cursors and for the admitted slew bisection.  The
    denominator always contains the full configured horizon, so clamping at a
    terminal target naturally decays the code instead of renormalizing the
    remaining rotation.

    Returns ``(code, validity)``.  ``zero_ablation`` changes only the emitted
    code; validity still certifies the real hidden preview so malformed target
    streams cannot be concealed by the ablation.
    """
    dt = jnp.asarray(control_dt_s, dtype=jnp.float32)
    horizon = jnp.asarray(horizon_steps, dtype=jnp.int32)
    rate_rad_s = jnp.deg2rad(jnp.asarray(rate_limit_deg_s, dtype=jnp.float32))
    scale = jnp.asarray(code_scale, dtype=jnp.float32)
    imminent = stream_target_batch(cursor, padded_targets, lengths, bounds, control_dt_s).next
    initial_valid = jnp.all(jnp.isfinite(imminent), axis=-1)

    def advance_preview(_: int, carry: tuple[StreamCursor, Array, Array]) -> tuple[StreamCursor, Array, Array]:
        (preview_cursor, previous_target, path_valid) = carry
        (next_cursor, next_targets) = advance_stream_cursor(preview_cursor, padded_targets, lengths, bounds, control_dt_s)
        following_target = next_targets.next
        edge_log = _relative_so3_log(_hpr_to_quaternion_nb(previous_target[..., :3]), _hpr_to_quaternion_nb(following_target[..., :3]))
        edge_limit = dt * rate_rad_s
        edge_valid = jnp.all(jnp.isfinite(following_target), axis=-1) & jnp.all(jnp.isfinite(edge_log), axis=-1) & (jnp.linalg.norm(edge_log, axis=-1) <= edge_limit + np.float32(1e-05))
        return (next_cursor, following_target, path_valid & edge_valid)
    (_, endpoint, path_valid) = jax.lax.fori_loop(0, horizon, advance_preview, (cursor, imminent, initial_valid))
    rotation_vector = _relative_so3_log(_hpr_to_quaternion_nb(imminent[..., :3]), _hpr_to_quaternion_nb(endpoint[..., :3]))
    identical_endpoint_frame = jnp.all(imminent[..., :3] == endpoint[..., :3], axis=-1)
    rotation_vector = jnp.where(identical_endpoint_frame[..., None], jnp.zeros_like(rotation_vector), rotation_vector)
    configuration_valid = (horizon > 0) & jnp.isfinite(dt) & (dt > np.float32(0.0)) & jnp.isfinite(rate_rad_s) & (rate_rad_s > np.float32(0.0)) & jnp.isfinite(scale) & (scale > np.float32(0.0))
    denominator = jnp.where(configuration_valid, horizon.astype(jnp.float32) * dt * rate_rad_s, jnp.asarray(jnp.nan, dtype=jnp.float32))
    raw_code = scale * rotation_vector / denominator
    validity = configuration_valid & path_valid & future_twist_preview_semantic_validity(raw_code)
    code = jnp.where(jnp.asarray(zero_ablation, dtype=jnp.bool_), jnp.zeros_like(raw_code), raw_code)
    return (code, validity)

_uncached_unwrap = unwrap_padded_target_bank
_unwrap_cache = None
def unwrap_padded_target_bank(padded_targets,lengths):
    global _unwrap_cache
    if _unwrap_cache is not None and _unwrap_cache[0] is padded_targets and _unwrap_cache[1] is lengths:
        return _unwrap_cache[2]
    result=_uncached_unwrap(padded_targets,lengths)
    result.setflags(write=False)
    _unwrap_cache=(padded_targets,lengths,result)
    return result
