"""Private coordinate bridge; not a deployed runtime or aircraft pass claim."""
import math
from types import FunctionType
import numpy as np
import vertical_chart
import guidance as base
import frame_math as frames

SCHEMA='planax.a4.return_gate.private_lifted_pitch_guidance.v1'
REPRESENTATION_LIMIT_RAD=4.*math.pi


def principal_hpr(hpr):
    value=np.asarray(hpr,np.float64)
    if abs(value[1])<=math.pi/2.:
        return value.copy()
    return np.asarray(frames.quaternion_to_euler(frames.quaternion_from_euler(*value)),np.float64)


def validate_inputs(positions,phase,targets):
    values=np.asarray(targets)
    if values.ndim!=2 or values.shape[1]!=4:
        return base.validate_a4_guidance_inputs(positions,phase,targets)
    if not np.isfinite(values).all() or np.any(np.abs(values[:,1])>REPRESENTATION_LIMIT_RAD):
        raise ValueError('invalid finite lifted-pitch representation')
    if np.all(np.abs(values[:,1])<=math.pi/2.+1.e-6):
        return base.validate_a4_guidance_inputs(positions,phase,targets)
    canonical=values.astype(np.float64,copy=True)
    for i in range(len(values)):canonical[i,:3]=principal_hpr(values[i,:3])
    pos,arc,_=base.validate_a4_guidance_inputs(positions,phase,canonical)
    return pos,arc,values.astype(np.float64,copy=True)


_original_init=base.A4ClosedLoopGuidance.__init__
_private_init=FunctionType(_original_init.__code__,dict(_original_init.__globals__,
    validate_a4_guidance_inputs=validate_inputs),_original_init.__name__,
    _original_init.__defaults__,_original_init.__closure__)


class LiftedPitchGuidance(base.A4ClosedLoopGuidance):
    def __init__(self,reference_positions_neu_m,reference_phase_m,nominal_target_hprv,config=None):
        _private_init(self,reference_positions_neu_m,reference_phase_m,nominal_target_hprv,config)
        self.uses_lifted_pitch=bool(np.any(np.abs(self.nominal_target_hprv[:,1])>math.radians(85.)))

    def _nominal_at(self,phase_s_m):
        value=super()._nominal_at(phase_s_m)
        if self.uses_lifted_pitch:value[:3]=principal_hpr(value[:3])
        return value

    def _slew(self,desired):
        if not self.uses_lifted_pitch:return super()._slew(desired)
        if self._previous_target is None:raise RuntimeError('guidance must be reset before its first step')
        value=np.asarray(desired,np.float64).copy()
        value[:3]=vertical_chart.lifted_euler(frames.quaternion_from_euler(*value[:3]),self._previous_target[:3])
        return super()._slew(value)


def validate_stream(targets,params):
    from research.projects.a4_geometry_energy_stream_compiler import a4_policy_rollout as original
    values=np.asarray(targets)
    if values.ndim!=2 or values.shape[1]!=4:
        return original.validate_precompiled_hprv_stream(targets,params)
    if np.all(np.abs(values[:,1])<=np.deg2rad(params.stream_pitch_limit_deg)+1.e-6):
        return original.validate_precompiled_hprv_stream(targets,params)
    # The original validator still checks every other channel, type, shape,
    # speed bound and padding invariant. Pitch-specific checks are explicit.
    proxy=values.copy();proxy[:,1]=0.
    checked=original.validate_precompiled_hprv_stream(proxy,params)
    result=values.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result[:,1])>REPRESENTATION_LIMIT_RAD+1.e-6):
        raise ValueError('lifted pitch exceeds its coordinate representation bound')
    dt=params.agent_interaction_steps/params.sim_freq
    if np.any(np.abs(np.diff(result[:,1]))>np.deg2rad(params.stream_pitch_slew_deg_s)*dt+2.e-6):
        raise ValueError('lifted pitch slew exceeds original executable bound')
    quaternion=[frames.quaternion_from_euler(*row[:3]) for row in result]
    steps=[frames._unit_quaternion_distance(a,b) for a,b in zip(quaternion[:-1],quaternion[1:])]
    if max(steps,default=0.)>math.radians(50.)*dt+2.e-6:
        raise ValueError('lifted stream exceeds original physical attitude slew')
    if not np.array_equal(result[:,[0,2,3]],checked[:,[0,2,3]]):
        raise ValueError('nonpitch channels changed at private validation boundary')
    return result
