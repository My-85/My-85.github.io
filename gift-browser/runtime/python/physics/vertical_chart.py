from __future__ import annotations
import math
import numpy as np
import frame_math as original
from types import SimpleNamespace
def _require(ok,message):
    if not ok: raise ValueError(message)
protocol=SimpleNamespace(require=_require)

def winding_near(value, reference):
    return float(value) + 2.0 * math.pi * math.floor((float(reference) - float(value) + math.pi) / (2.0 * math.pi))

def lifted_euler(quaternion, reference_hpr_rad):
    reference = np.asarray(reference_hpr_rad, dtype=np.float64)
    principal = original.quaternion_to_lifted_euler_near_reference(quaternion, reference)
    first = principal.copy()
    first[1] = winding_near(first[1], reference[1])
    alternate = np.asarray([winding_near(principal[0] + math.pi, reference[0]), winding_near(math.pi - principal[1], reference[1]), winding_near(principal[2] + math.pi, reference[2])])
    result = first if np.sum((first - reference) ** 2) <= np.sum((alternate - reference) ** 2) else alternate
    reconstructed = original.quaternion_from_euler(*result)
    protocol.require(original._quaternion_distance(reconstructed, np.asarray(quaternion)) <= 2e-07, 'two-chart Euler lift did not preserve physical attitude')
    return result
