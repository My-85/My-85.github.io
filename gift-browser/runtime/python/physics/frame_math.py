from __future__ import annotations
import math
import numpy as np
_EPS=1.e-12

def wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)

def quaternion_from_euler(heading: float, pitch: float, roll: float) -> np.ndarray:
    (cy, sy) = (math.cos(0.5 * heading), math.sin(0.5 * heading))
    (cp, sp) = (math.cos(0.5 * pitch), math.sin(0.5 * pitch))
    (cr, sr) = (math.cos(0.5 * roll), math.sin(0.5 * roll))
    return np.asarray([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float64)

def quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    (w, x, y, z) = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (wrap_pi(heading), float(pitch), wrap_pi(roll))

def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Unit quaternion to body-to-NED rotation matrix."""
    (w, x, y, z) = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= _EPS:
        raise ValueError('quaternion norm must be positive')
    (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)], [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)], [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float64)

class ForceFrameRollTubeError(ValueError):
    """Base class for malformed or uncertifiable roll-tube requests."""

def _lift_angle(value: float, reference: float) -> float:
    return float(reference + wrap_pi(float(value) - float(reference)))

def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_unit = np.asarray(first, dtype=np.float64)
    second_unit = np.asarray(second, dtype=np.float64)
    first_unit = first_unit / max(float(np.linalg.norm(first_unit)), _EPS)
    second_unit = second_unit / max(float(np.linalg.norm(second_unit)), _EPS)
    chord = min(float(np.linalg.norm(first_unit - second_unit)), float(np.linalg.norm(first_unit + second_unit)))
    return float(4.0 * math.asin(float(np.clip(0.5 * chord, 0.0, 1.0))))

def quaternion_to_lifted_euler_near_reference(quaternion: np.ndarray, reference_hpr_rad: np.ndarray) -> np.ndarray:
    """Convert one attitude without discarding roll at the pitch singularity.

    The ordinary inverse Euler formula evaluates two ``atan2(0, 0)`` terms at
    exactly vertical pitch and can return an H/P/R triple that does not encode
    the input quaternion.  At that singularity only ``heading-roll`` (nose up)
    or ``heading+roll`` (nose down) is observable.  Select the member of that
    equivalence class nearest the supplied lifted H/P/R reference.
    """
    rotation = quaternion_to_rotation(quaternion)
    reference = np.asarray(reference_hpr_rad, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ForceFrameRollTubeError('Euler lift reference must be one finite H/P/R')
    sine_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    cosine_pitch = math.sqrt(max(1.0 - sine_pitch * sine_pitch, 0.0))
    if cosine_pitch > 1e-07:
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        heading = _lift_angle(heading, reference[0])
        roll = _lift_angle(roll, reference[2])
    elif sine_pitch >= 0.0:
        delta_wrapped = math.atan2(rotation[1, 2], rotation[0, 2])
        reference_delta = float(reference[0] - reference[2])
        delta = reference_delta + wrap_pi(delta_wrapped - reference_delta)
        heading = 0.5 * (float(reference[0] + reference[2]) + delta)
        roll = 0.5 * (float(reference[0] + reference[2]) - delta)
        pitch = 0.5 * math.pi
    else:
        sigma_wrapped = math.atan2(-rotation[1, 2], rotation[1, 1])
        reference_sigma = float(reference[0] + reference[2])
        sigma = reference_sigma + wrap_pi(sigma_wrapped - reference_sigma)
        heading = 0.5 * (float(reference[0] - reference[2]) + sigma)
        roll = 0.5 * (float(reference[2] - reference[0]) + sigma)
        pitch = -0.5 * math.pi
    result = np.asarray([heading, pitch, roll], dtype=np.float64)
    reconstructed = quaternion_from_euler(*result)
    if _quaternion_distance(reconstructed, np.asarray(quaternion)) > 2e-07:
        raise ForceFrameRollTubeError('reference-aware Euler conversion did not preserve the attitude')
    return result
