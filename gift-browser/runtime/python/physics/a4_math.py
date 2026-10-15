from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Dict
import numpy as np
from numpy_compat import jnp, jax, at_set, ReplaceMixin
Array = np.ndarray
from aircraft_specs import A4_SPEC

def atmos(alt, vt):
    rho0 = np.float32(0.002377)
    tfac = 1 - np.float32(7.03e-06) * alt
    temp = np.float32(519.0) * tfac
    temp = (alt >= np.float32(35000.0)) * np.float32(390.0) + (alt < np.float32(35000.0)) * temp
    rho = rho0 * jnp.pow(tfac, np.float32(4.14))
    mach = vt / jnp.sqrt(np.float32(1.4) * np.float32(1716.3) * temp)
    qbar = np.float32(0.5) * rho * jnp.pow(vt, 2)
    ps = np.float32(1715.0) * rho * temp
    ps = (ps == 0) * np.float32(1715.0) + (ps != 0) * ps
    return (mach, qbar, ps)

def accels(roll, pitch, alpha, beta, vt, alpha_dot, beta_dot, vt_dot, P, Q, R):
    grav = np.float32(32.174)
    sina = jnp.sin(alpha)
    cosa = jnp.cos(alpha)
    sinb = jnp.sin(beta)
    cosb = jnp.cos(beta)
    vel_u = vt * cosb * cosa
    vel_v = vt * sinb
    vel_w = vt * cosb * sina
    u_dot = cosb * cosa * vt_dot - vt * sinb * cosa * beta_dot - vt * cosb * sina * alpha_dot
    v_dot = sinb * vt_dot + vt * cosb * beta_dot
    w_dot = cosb * sina * vt_dot - vt * sinb * sina * beta_dot + vt * cosb * cosa * alpha_dot
    nx_cg = -np.float32(1.0) / grav * (u_dot + Q * vel_w - R * vel_v) - jnp.sin(pitch)
    ny_cg = -np.float32(1.0) / grav * (v_dot + R * vel_u - P * vel_w) + jnp.cos(pitch) * jnp.sin(roll)
    nz_cg = -np.float32(1.0) / grav * (w_dot + P * vel_v - Q * vel_u) + jnp.cos(pitch) * jnp.cos(roll)
    return (nx_cg, ny_cg, nz_cg)

def quaternion_to_rpy(q0, q1, q2, q3):
    """
    将四元数(q0,q1,q2,q3)转换为欧拉角(roll, pitch, yaw).
    假定:
       - q0为实部, (q1,q2,q3)为空间部
       - 机体系 'body' -> 'inertial' 的Z-Y-X顺序
       - roll绕 x, pitch绕 y, yaw绕 z
    """
    '\n        Calculate attitude angle using quaternion q_{NED}^{Body}, return Eular angle in unit degree.\n        注意：这里传入的四元数是 q_{NED}^{Body}，即从机体系到NED系的四元数。\n        roll    Range [-180,180)\n        pitch   Range [-90,90]\n        yaw     Range [-180,180)\n    '
    sinr_cosp = np.float32(2.0) * (q0 * q1 + q2 * q3)
    cosr_cosp = np.float32(1.0) - np.float32(2.0) * (q1 * q1 + q2 * q2)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.float32(2.0) * (q0 * q2 - q3 * q1)
    pitch = jnp.where(jnp.abs(sinp) >= np.float32(1.0), jnp.sign(sinp) * (jnp.pi / np.float32(2.0)), jnp.arcsin(sinp))
    siny_cosp = np.float32(2.0) * (q0 * q3 + q1 * q2)
    cosy_cosp = np.float32(1.0) - np.float32(2.0) * (q2 * q2 + q3 * q3)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)

def _interp1(x, xp, fp):
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)
    x_clipped = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.clip(jnp.searchsorted(xp, x_clipped, side='right') - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    t = (x_clipped - x0) / (x1 - x0 + np.float32(1e-06))
    return y0 + t * (y1 - y0)

def _map_bipolar_action(command, neg_limit, pos_limit):
    command = jnp.clip(command, -np.float32(1.0), np.float32(1.0))
    return jnp.where(command >= np.float32(0.0), command * pos_limit, -command * neg_limit)

def _first_order(previous, command, dt, tau):
    coeff = jnp.exp(-dt / tau)
    return coeff * previous + (np.float32(1.0) - coeff) * command

def _a4_thrust_lapse(mach, qbar, vt):
    rho = np.float32(2.0) * qbar / (vt * vt + np.float32(1e-06))
    sigma = jnp.clip(rho / np.float32(0.002377), np.float32(0.05), np.float32(1.5))
    return jnp.clip(jnp.power(sigma, np.float32(0.7)) * (np.float32(1.0) + np.float32(0.08) * mach), np.float32(0.2), np.float32(1.25))

def _a4_coefficients(alpha, beta, mach, p, q, r, elevator, aileron, rudder, speedbrake, vt):
    cl_alpha = _interp1(alpha, jnp.array([-np.float32(0.2), np.float32(0.0), np.float32(0.26), np.float32(0.6)]), jnp.array([-np.float32(0.62), np.float32(0.08), np.float32(1.0), np.float32(0.448)]))
    cd0 = _interp1(alpha, jnp.array([-np.float32(1.57), -np.float32(0.26), np.float32(0.0), np.float32(0.26), np.float32(1.57)]), jnp.array([np.float32(1.5), np.float32(0.027), np.float32(0.021), np.float32(0.027), np.float32(1.5)]))
    cd_mach = _interp1(mach, jnp.array([np.float32(0.0), np.float32(0.81), np.float32(1.1), np.float32(1.8)]), jnp.array([np.float32(0.0), np.float32(0.0), np.float32(0.023), np.float32(0.015)]))
    cl = cl_alpha + np.float32(0.2) * elevator
    cd = cd0 + np.float32(0.09) * cl * cl + cd_mach + np.float32(0.021) * jnp.clip(speedbrake, np.float32(0.0), np.float32(1.0)) + _interp1(beta, jnp.array([-np.float32(1.57), -np.float32(0.26), np.float32(0.0), np.float32(0.26), np.float32(1.57)]), jnp.array([np.float32(1.23), np.float32(0.05), np.float32(0.0), np.float32(0.05), np.float32(1.23)])) + np.float32(0.04) * jnp.abs(elevator)
    cx = -cd * jnp.cos(alpha) + cl * jnp.sin(alpha)
    cz = -cd * jnp.sin(alpha) - cl * jnp.cos(alpha)
    b_2v = A4_SPEC.wingspan_ft / (np.float32(2.0) * vt + np.float32(1e-06))
    c_2v = A4_SPEC.chord_ft / (np.float32(2.0) * vt + np.float32(1e-06))
    clda = _interp1(mach, jnp.array([np.float32(0.0), np.float32(2.0)]), jnp.array([np.float32(0.11), np.float32(0.037)]))
    cmde = _interp1(mach, jnp.array([np.float32(0.0), np.float32(2.0)]), jnp.array([-np.float32(0.5), -np.float32(0.2)]))
    cy = -np.float32(1.0) * beta
    cl_roll = -np.float32(0.1) * beta - np.float32(0.4) * b_2v * p + np.float32(0.15) * b_2v * r + clda * aileron + np.float32(0.01) * rudder
    cm = -np.float32(0.38) * alpha + cmde * elevator - np.float32(3.6) * c_2v * q
    cn = np.float32(0.12) * beta - np.float32(0.15) * b_2v * r - np.float32(0.1) * rudder
    return (cx, cy, cz, cl_roll, cm, cn)

def nlplant_a4(xu, aero_scale=np.float32(1.0), mass_scale=np.float32(1.0), inertia_scale=np.float32(1.0), wind_u=np.float32(0.0), wind_v=np.float32(0.0), wind_w=np.float32(0.0)):
    xdot = jnp.zeros_like(xu)
    g = np.float32(32.17)
    mass = A4_SPEC.mass_slug * mass_scale
    wing_area = A4_SPEC.wing_area_ft2
    wingspan = A4_SPEC.wingspan_ft
    chord = A4_SPEC.chord_ft
    jx = A4_SPEC.jx_slug_ft2 * inertia_scale
    jy = A4_SPEC.jy_slug_ft2 * inertia_scale
    jz = A4_SPEC.jz_slug_ft2 * inertia_scale
    jxz = A4_SPEC.jxz_slug_ft2 * inertia_scale
    alt = xu[2]
    vt = jnp.maximum(xu[6], np.float32(0.01))
    alpha = xu[7]
    beta = xu[8]
    p = xu[9]
    q = xu[10]
    r = xu[11]
    q0 = xu[12]
    q1 = xu[13]
    q2 = xu[14]
    q3 = xu[15]
    q0sq = q0 ** 2
    q1sq = q1 ** 2
    q2sq = q2 ** 2
    q3sq = q3 ** 2
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q3 = q2 * q3
    thrust = xu[16]
    elevator = xu[17]
    aileron = xu[18]
    rudder = xu[19]
    speedbrake = xu[21]
    sin_alpha = jnp.sin(alpha)
    cos_alpha = jnp.cos(alpha)
    sin_beta = jnp.sin(beta)
    cos_beta = jnp.cos(beta)
    u = vt * cos_alpha * cos_beta
    v = vt * sin_beta
    w = vt * sin_alpha * cos_beta
    xdot = at_set(xdot, 0, (q0sq + q1sq - q2sq - q3sq) * u + np.float32(2.0) * (q1q2 + q0q3) * v + np.float32(2.0) * (q1q3 - q0q2) * w)
    xdot = at_set(xdot, 1, np.float32(2.0) * (q1q2 - q0q3) * u + (q0sq - q1sq + q2sq - q3sq) * v + np.float32(2.0) * (q2q3 + q0q1) * w)
    xdot = at_set(xdot, 2, -(np.float32(2.0) * (q1q3 + q0q2) * u + np.float32(2.0) * (q2q3 - q0q1) * v + (q0sq - q1sq - q2sq + q3sq) * w))
    xdot = at_set(xdot, 3, np.float32(0.0))
    xdot = at_set(xdot, 4, np.float32(0.0))
    xdot = at_set(xdot, 5, np.float32(0.0))
    u_air = u - wind_u
    v_air = v - wind_v
    w_air = w - wind_w
    vt_air = jnp.maximum(jnp.sqrt(u_air * u_air + v_air * v_air + w_air * w_air), np.float32(0.01))
    alpha_eff = alpha + jnp.arctan2(w_air, u_air) - jnp.arctan2(w, u)
    beta_eff = beta + jnp.arcsin(jnp.clip(v_air / vt_air, -np.float32(1.0), np.float32(1.0))) - jnp.arcsin(jnp.clip(v / vt, -np.float32(1.0), np.float32(1.0)))
    (mach, qbar, _) = atmos(alt, vt_air)
    (cx, cy, cz, cl, cm, cn) = _a4_coefficients(alpha_eff, beta_eff, mach, p, q, r, elevator, aileron, rudder, speedbrake, vt_air)
    udot = r * v - q * w + g * np.float32(2.0) * (q1q3 + q0q2) + qbar * wing_area * cx * aero_scale / mass + thrust / mass
    vdot = p * w - r * u + g * np.float32(2.0) * (q2q3 - q0q1) + qbar * wing_area * cy * aero_scale / mass
    wdot = q * u - p * v + g * (q0sq - q1sq - q2sq + q3sq) + qbar * wing_area * cz * aero_scale / mass
    xdot = at_set(xdot, 6, (u * udot + v * vdot + w * wdot) / (vt + np.float32(1e-06)))
    xdot = at_set(xdot, 7, (u * wdot - w * udot) / (u * u + w * w + np.float32(1e-06)))
    xdot = at_set(xdot, 8, (vdot * vt - v * xdot[6]) / (vt * vt * cos_beta + np.float32(1e-06)))
    l_tot = cl * qbar * wing_area * wingspan * aero_scale
    m_tot = cm * qbar * wing_area * chord * aero_scale
    n_tot = cn * qbar * wing_area * wingspan * aero_scale
    denom = jx * jz - jxz * jxz + np.float32(1e-06)
    xdot = at_set(xdot, 9, (jz * l_tot + jxz * n_tot - (jz * (jz - jy) + jxz * jxz) * q * r + jxz * (jx - jy + jz) * p * q) / denom)
    xdot = at_set(xdot, 10, (m_tot + (jz - jx) * p * r - jxz * (p * p - r * r)) / jy)
    xdot = at_set(xdot, 11, (jx * n_tot + jxz * l_tot + (jx * (jx - jy) + jxz * jxz) * p * q - jxz * (jx - jy + jz) * q * r) / denom)
    xdot = at_set(xdot, 12, np.float32(0.5) * (p * q1 + q * q2 + r * q3))
    xdot = at_set(xdot, 13, np.float32(0.5) * (-p * q0 + r * q2 - q * q3))
    xdot = at_set(xdot, 14, np.float32(0.5) * (-q * q0 - r * q1 + p * q3))
    xdot = at_set(xdot, 15, np.float32(0.5) * (-r * q0 + q * q1 - p * q2))
    return xdot

def update(state: FighterPlaneState, action: FighterPlaneControlState, dt: float) -> FighterPlaneState:
    x = jnp.hstack((state.north / np.float32(0.3048), state.east / np.float32(0.3048), state.altitude / np.float32(0.3048), state.roll, state.pitch, state.yaw, state.vt / np.float32(0.3048), state.alpha, state.beta, state.P, state.Q, state.R, state.q0, state.q1, state.q2, state.q3))
    (mach, qbar, _) = atmos(x[2], jnp.maximum(x[6], np.float32(0.01)))
    thrust_cmd = jnp.clip(action.throttle, np.float32(0.0), np.float32(1.0)) * A4_SPEC.max_thrust_lbf * _a4_thrust_lapse(mach, qbar, x[6])
    thrust = _first_order(state.T, thrust_cmd, dt, A4_SPEC.throttle_tau_s)
    elevator = _first_order(state.el, _map_bipolar_action(action.elevator, A4_SPEC.elevator_min_rad, A4_SPEC.elevator_max_rad), dt, A4_SPEC.surface_tau_s)
    aileron = _first_order(state.ail, jnp.clip(action.aileron, -np.float32(1.0), np.float32(1.0)) * A4_SPEC.aileron_max_rad, dt, A4_SPEC.surface_tau_s)
    rudder = _first_order(state.rud, jnp.clip(action.rudder, -np.float32(1.0), np.float32(1.0)) * A4_SPEC.rudder_max_rad, dt, A4_SPEC.surface_tau_s)
    lef = jnp.asarray(np.float32(0.0), dtype=x.dtype)
    speedbrake = _first_order(state.sb, jnp.clip(action.speed_brake, A4_SPEC.speedbrake_min, A4_SPEC.speedbrake_max), dt, A4_SPEC.speedbrake_tau_s)
    xu = jnp.hstack((x, jnp.hstack((thrust, elevator, aileron, rudder, lef, speedbrake))))
    xdot = nlplant_a4(xu, aero_scale=state.aero_scale, mass_scale=state.mass_scale, inertia_scale=state.inertia_scale, wind_u=state.wind_body_u / np.float32(0.3048), wind_v=state.wind_body_v / np.float32(0.3048), wind_w=state.wind_body_w / np.float32(0.3048))
    (nx_cg, ny_cg, nz_cg) = accels(xu[3], xu[4], xu[7], xu[8], xu[6], xdot[7], xdot[8], xdot[6], xu[9], xu[10], xu[11])
    new_x = x + xdot[:16] * dt
    new_q0 = new_x[12]
    new_q1 = new_x[13]
    new_q2 = new_x[14]
    new_q3 = new_x[15]
    norm_q = jnp.sqrt(new_q0 ** 2 + new_q1 ** 2 + new_q2 ** 2 + new_q3 ** 2) + np.float32(1e-06)
    new_q0 = new_q0 / norm_q
    new_q1 = new_q1 / norm_q
    new_q2 = new_q2 / norm_q
    new_q3 = new_q3 / norm_q
    dot = state.q0 * new_q0 + state.q1 * new_q1 + state.q2 * new_q2 + state.q3 * new_q3
    sign = jnp.where(dot < np.float32(0.0), -np.float32(1.0), np.float32(1.0))
    new_q0 = new_q0 * sign
    new_q1 = new_q1 * sign
    new_q2 = new_q2 * sign
    new_q3 = new_q3 * sign
    (roll, pitch, yaw) = quaternion_to_rpy(new_q0, -new_q1, -new_q2, -new_q3)
    new_state = state.replace(north=jnp.nan_to_num(new_x[0] * np.float32(0.3048), nan=np.float32(0.0)), east=jnp.nan_to_num(new_x[1] * np.float32(0.3048), nan=np.float32(0.0)), altitude=jnp.nan_to_num(new_x[2] * np.float32(0.3048), nan=np.float32(0.0)), roll=jnp.nan_to_num(roll, nan=np.float32(0.0)), pitch=jnp.nan_to_num(pitch, nan=np.float32(0.0)), yaw=jnp.nan_to_num(yaw, nan=np.float32(0.0)), vel_x=jnp.nan_to_num(xdot[0] * np.float32(0.3048), nan=np.float32(0.0)), vel_y=jnp.nan_to_num(xdot[1] * np.float32(0.3048), nan=np.float32(0.0)), vel_z=jnp.nan_to_num(xdot[2] * np.float32(0.3048), nan=np.float32(0.0)), vt=jnp.nan_to_num(new_x[6] * np.float32(0.3048), nan=np.float32(0.0)), alpha=jnp.nan_to_num(new_x[7], nan=np.float32(0.0)), beta=jnp.nan_to_num(new_x[8], nan=np.float32(0.0)), P=jnp.nan_to_num(new_x[9], nan=np.float32(0.0)), Q=jnp.nan_to_num(new_x[10], nan=np.float32(0.0)), R=jnp.nan_to_num(new_x[11], nan=np.float32(0.0)), q0=jnp.nan_to_num(new_q0, nan=np.float32(1.0)), q1=jnp.nan_to_num(new_q1, nan=np.float32(0.0)), q2=jnp.nan_to_num(new_q2, nan=np.float32(0.0)), q3=jnp.nan_to_num(new_q3, nan=np.float32(0.0)), T=jnp.nan_to_num(thrust, nan=np.float32(0.0)), el=jnp.nan_to_num(elevator, nan=np.float32(0.0)), ail=jnp.nan_to_num(aileron, nan=np.float32(0.0)), rud=jnp.nan_to_num(rudder, nan=np.float32(0.0)), lef=jnp.nan_to_num(lef, nan=np.float32(0.0)), sb=jnp.nan_to_num(speedbrake, nan=np.float32(0.0)), ax=jnp.nan_to_num(nx_cg, nan=np.float32(0.0)), ay=jnp.nan_to_num(ny_cg, nan=np.float32(0.0)), az=jnp.nan_to_num(nz_cg, nan=np.float32(0.0)), aero_scale=state.aero_scale, mass_scale=state.mass_scale, inertia_scale=state.inertia_scale, wind_body_u=state.wind_body_u, wind_body_v=state.wind_body_v, wind_body_w=state.wind_body_w)
    mask = state.is_alive | state.is_locked
    return jax.lax.cond(mask, lambda : new_state, lambda : state)
