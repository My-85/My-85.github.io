from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Dict
import numpy as np
from numpy_compat import jnp, jax, at_set, ReplaceMixin
Array = np.ndarray

def check_collision(state: BasePlaneState, agent_id, R=20):
    alive = state.is_alive | state.is_locked
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((state.north, state.east, state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = at_set(distance, agent_id, jnp.finfo(jnp.float32).max)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    done = jnp.any(distance < R)
    return done

def check_extreme_state(state: BasePlaneState, agent_id, rotation_limit=np.float32(1000.0)):
    (P, Q, R) = (state.P[agent_id], state.Q[agent_id], state.R[agent_id])
    done = jnp.sqrt(P ** 2 + Q ** 2 + R ** 2) > rotation_limit
    return done

def check_high_speed(state: BasePlaneState, agent_id, max_velocity=3):
    velocity = state.vt[agent_id] / 340
    done = velocity > max_velocity
    return done

def check_low_speed(state: BasePlaneState, agent_id, min_velocity=np.float32(0.01)):
    velocity = state.vt[agent_id] / 340
    done = velocity < min_velocity
    return done

def check_high_altitude(state: BasePlaneState, agent_id, altitude_limit=np.float32(1000000000.0)):
    altitude = state.altitude[agent_id]
    done = altitude > altitude_limit
    return done

def check_low_altitude(state: BasePlaneState, agent_id, altitude_limit=np.float32(2500.0)):
    altitude = state.altitude[agent_id]
    done = altitude < altitude_limit
    return done

def check_overload(state: BasePlaneState, agent_id, max_overload=np.float32(10.0)):
    mask1 = jnp.abs(state.ax[agent_id]) >= max_overload
    mask2 = jnp.abs(state.ay[agent_id]) >= max_overload
    mask3 = jnp.abs(state.az[agent_id]) >= max_overload
    done = mask1 | mask2 | mask3
    return done

def check_crashed(state: BasePlaneState, agent_id):
    mask1 = check_collision(state, agent_id)
    mask2 = check_extreme_state(state, agent_id)
    mask3 = check_high_speed(state, agent_id)
    mask4 = check_low_speed(state, agent_id)
    mask5 = check_low_altitude(state, agent_id)
    mask6 = check_overload(state, agent_id)
    mask7 = check_high_altitude(state, agent_id)
    crashed = mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7
    return crashed
