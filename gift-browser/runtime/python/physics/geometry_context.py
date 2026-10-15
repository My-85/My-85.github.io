"""Optional explicit reference-geometry input, not an aircraft observation edit.

Only the requested path and currently observed aircraft state are used. No
route identity, split, teacher label, future aircraft state, or wall time is
an input. This frontend is a candidate for gating; the running paired repair
fits do not use it. The frozen21-D source observation ABI stays unchanged.
"""
import numpy as np

HORIZONS_S=(0.,2.,4.,8.,16.,32.)
DIM=len(HORIZONS_S)*6


def encode(reference,phase,*,cursor,position,yaw,speed):
    reference=np.asarray(reference,np.float64);phase=np.asarray(phase,np.float64)
    position=np.asarray(position,np.float64)
    if reference.ndim!=2 or reference.shape[1]!=3 or len(reference)<2 or phase.shape!=(len(reference),):
        raise ValueError('ordered NEU reference and arc coordinates required')
    if position.shape!=(3,) or not all(np.isfinite(x).all() for x in (reference,phase,position)):
        raise ValueError('finite physical/reference inputs required')
    if not np.isfinite([cursor,yaw,speed]).all() or speed<=0 or np.any(np.diff(phase)<=0):
        raise ValueError('finite measured state and increasing arc coordinates required')
    # The native stream cursor is a fractional sample index, not arc metres.
    s=np.interp(float(cursor),np.arange(len(phase)),phase)
    stations=np.clip(s+speed*np.asarray(HORIZONS_S),phase[0],phase[-1])
    at=np.stack([np.interp(stations,phase,reference[:,k]) for k in range(3)],-1)
    tangent=_cached_tangent(reference,phase)
    tangent=np.stack([np.interp(stations,phase,tangent[:,k]) for k in range(3)],-1)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1.e-9)
    # Horizontal heading rotation and translation leave the input invariant;
    # gravity's vertical direction is deliberately NOT made rotationally invariant.
    c,sn=np.cos(yaw),np.sin(yaw)
    rotation=np.asarray([[c,-sn,0.],[sn,c,0.],[0.,0.,1.]])
    relative=(at-position)@rotation
    relative/=np.maximum(250.,speed*np.asarray(HORIZONS_S))[:,None]
    return np.concatenate((np.clip(relative,-10.,10.),tangent@rotation),-1).astype(np.float32).reshape(DIM)


def archive_inputs(raw):
    n=len(raw['action_indices'])
    if any(len(raw[k])!=n+1 for k in ('actual_hprv','actual_position_nea_m')):
        raise ValueError('pre-action physical samples including terminal required')
    return np.stack([encode(raw['reference_neu_m'],raw['reference_phase_m'],
        cursor=raw['stream_phase_before'][i],position=raw['actual_position_nea_m'][i],
        yaw=raw['actual_hprv'][i,0],speed=raw['actual_hprv'][i,3]) for i in range(n)])

_tangent_cache=None
def _cached_tangent(reference,phase):
    global _tangent_cache
    if _tangent_cache is not None and _tangent_cache[0] is reference and _tangent_cache[1] is phase:
        return _tangent_cache[2]
    result=np.gradient(reference,phase,axis=0)
    result.setflags(write=False)
    _tangent_cache=(reference,phase,result)
    return result
