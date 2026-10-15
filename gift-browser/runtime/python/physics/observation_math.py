from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Dict
import numpy as np
from numpy_compat import jnp, jax, at_set, ReplaceMixin
Array = np.ndarray

@dataclass(frozen=True)
class Heading_Pitch_V_TaskState(ReplaceMixin):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_vt: ArrayLike
    target_roll: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    level_selected: ArrayLike = 0

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(plane_state=env_state.plane_state, missile_state=env_state.missile_state, control_state=env_state.control_state, pre_rewards=env_state.pre_rewards, done=env_state.done, success=env_state.success, time=env_state.time, target_heading=extra_state[0], target_pitch=extra_state[1], target_roll=extra_state[2], target_vt=extra_state[3], last_check_time=env_state.time, heading_turn_counts=jnp.zeros_like(extra_state[0], dtype=jnp.int32), level_selected=jnp.zeros_like(extra_state[0], dtype=jnp.int32))

@dataclass(frozen=True)
class Heading_Pitch_V_TaskParams(ReplaceMixin):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = np.float32(20000.0)
    min_altitude: float = np.float32(2000.0)
    max_vt: float = np.float32(360.0)
    min_vt: float = np.float32(120.0)
    max_velocities_u_increment: float = np.float32(50.0)
    max_heading_increment: float = jnp.pi / 2
    max_pitch_increment: float = jnp.pi / 6
    max_roll_increment: float = jnp.pi / 2
    max_altitude_increment: float = np.float32(2100.0)
    safe_altitude: float = np.float32(4.0)
    danger_altitude: float = np.float32(3.5)
    noise_scale: float = np.float32(0.0)
    team_spacing: float = 15000
    safe_distance: float = 3000
    nz_limit: float = np.float32(9.0)
    r_nz_coef: float = np.float32(0.05)
    r_nz_clip: float = np.float32(5.0)
    nz_hard_cap: float = np.float32(15.0)

def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + np.float32(1e-09))

def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul(q1, q2):
    (w1, x1, y1, z1) = q1
    (w2, x2, y2, z2) = q2
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

def _quat_from_euler_nb(roll, pitch, yaw):
    """Build quaternion from Tait-Bryan (Z-Y-X) Euler angles.
    Returns q representing Body->NED."""
    (cr, sr) = (jnp.cos(np.float32(0.5) * roll), jnp.sin(np.float32(0.5) * roll))
    (cp, sp) = (jnp.cos(np.float32(0.5) * pitch), jnp.sin(np.float32(0.5) * pitch))
    (cy, sy) = (jnp.cos(np.float32(0.5) * yaw), jnp.sin(np.float32(0.5) * yaw))
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.array([qw, qx, qy, qz])

def _target_q_bn_from_heading_pitch(yaw_t, pitch_t, roll_t=np.float32(0.0)):
    q_nb = _quat_from_euler_nb(roll_t, pitch_t, yaw_t)
    return _quat_conj(q_nb)

def _quat_err_bn(q_curr_bn, yaw_t, pitch_t, roll_t):
    q_curr_bn = _quat_normalize(jnp.nan_to_num(q_curr_bn, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0)))
    q_tgt_bn = _quat_normalize(_target_q_bn_from_heading_pitch(yaw_t, pitch_t, roll_t))
    q_err = _quat_mul(q_tgt_bn, _quat_conj(q_curr_bn))
    q_err = jnp.where(q_err[0] < np.float32(0.0), -q_err, q_err)
    return q_err

def _rotate_ned_to_body(q_bn, v_n):
    q_bn = _quat_normalize(q_bn)
    p = jnp.array([np.float32(0.0), v_n[0], v_n[1], v_n[2]])
    qpq = _quat_mul(_quat_mul(q_bn, p), _quat_conj(q_bn))
    return qpq[1:]

def base_observation(self, state: Heading_Pitch_V_TaskState, params: Heading_Pitch_V_TaskParams) -> Dict[AgentName, chex.Array]:
    """
        21-dim observation:
        [0:3]   qv (quaternion error vector part)
        [3]     (vt - target_vt) / 340
        [4]     altitude / 5000
        [5]     vt / 340
        [6:9]   v_b (target direction in body frame)
        [9:12]  P, Q, R
        [12:14] sin/cos alpha
        [14:16] sin/cos beta
        [16:21] prev_action (throttle, elevator, aileron, rudder, speed_brake)  — normalized
        """
    B = self.num_agents
    q_curr = jnp.stack([jnp.nan_to_num(state.plane_state.q0, nan=np.float32(0.0)), jnp.nan_to_num(state.plane_state.q1, nan=np.float32(0.0)), jnp.nan_to_num(state.plane_state.q2, nan=np.float32(0.0)), jnp.nan_to_num(state.plane_state.q3, nan=np.float32(0.0))], axis=1)
    yaw_t = state.target_heading
    pitch_t = state.target_pitch
    roll_t = state.target_roll
    vt_tgt = state.target_vt

    def _err_row(q_row, yh, ph, rh):
        return _quat_err_bn(q_row, yh, ph, rh)
    q_err_batch = jax.vmap(_err_row, in_axes=(0, 0, 0, 0))(q_curr, yaw_t, pitch_t, roll_t)
    qv = jnp.clip(q_err_batch[:, 1:4], -np.float32(1.0), np.float32(1.0))
    (c_th, s_th) = (jnp.cos(yaw_t), jnp.sin(yaw_t))
    (c_ph, s_ph) = (jnp.cos(pitch_t), jnp.sin(pitch_t))
    v_n = jnp.stack([c_ph * c_th, c_ph * s_th, -s_ph], axis=1)
    v_b = jax.vmap(_rotate_ned_to_body, in_axes=(0, 0))(q_curr, v_n)
    v_b = jnp.clip(v_b, -np.float32(1.0), np.float32(1.0))
    altitude = state.plane_state.altitude
    vt = state.plane_state.vt
    alpha = state.plane_state.alpha
    beta = state.plane_state.beta
    (P, Q, R) = (state.plane_state.P, state.plane_state.Q, state.plane_state.R)
    norm_dvt = (vt - vt_tgt) / np.float32(340.0)
    norm_alt = altitude / np.float32(5000.0)
    norm_vt = vt / np.float32(340.0)
    (alpha_sin, alpha_cos) = (jnp.sin(alpha), jnp.cos(alpha))
    (beta_sin, beta_cos) = (jnp.sin(beta), jnp.cos(beta))
    cs = state.control_state
    prev_thr = jnp.nan_to_num(cs.throttle, nan=np.float32(0.0))
    prev_el = jnp.nan_to_num(cs.elevator, nan=np.float32(0.0))
    prev_ail = jnp.nan_to_num(cs.aileron, nan=np.float32(0.0))
    prev_rud = jnp.nan_to_num(cs.rudder, nan=np.float32(0.0))
    prev_sb = jnp.nan_to_num(jnp.where(cs.speed_brake > 0, cs.speed_brake, jnp.zeros_like(cs.speed_brake)), nan=np.float32(0.0))
    obs_mat = jnp.stack([qv[:, 0], qv[:, 1], qv[:, 2], norm_dvt, norm_alt, norm_vt, v_b[:, 0], v_b[:, 1], v_b[:, 2], P, Q, R, alpha_sin, alpha_cos, beta_sin, beta_cos, prev_thr, prev_el, prev_ail, prev_rud, prev_sb], axis=0)
    low_base = jnp.array([-np.float32(1.0), -np.float32(1.0), -np.float32(1.0), -np.float32(2.0), np.float32(0.0), np.float32(0.0), -np.float32(1.0), -np.float32(1.0), -np.float32(1.0), -np.float32(10.0), -np.float32(10.0), -np.float32(10.0), -np.float32(1.0), -np.float32(1.0), -np.float32(1.0), -np.float32(1.0)])
    high_base = jnp.array([np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(2.0), np.float32(5.0), np.float32(2.0), np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(10.0), np.float32(10.0), np.float32(10.0), np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(1.0)])
    low_act = jnp.array([np.float32(0.0), -np.float32(1.0), -np.float32(1.0), -np.float32(1.0), np.float32(0.0)])
    high_act = jnp.array([np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(1.0), np.float32(1.0)])
    low = jnp.concatenate([low_base, low_act]).reshape(-1, 1)
    high = jnp.concatenate([high_base, high_act]).reshape(-1, 1)
    obs_mat = jnp.clip(jnp.nan_to_num(obs_mat, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0)), low, high)
    return {agent: obs_mat[:, i] for (i, agent) in enumerate(self.agents)}
