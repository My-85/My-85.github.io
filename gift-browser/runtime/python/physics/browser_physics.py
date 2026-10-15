"""Entire geometry/observation/guidance/A4 flight runtime for browser NumPy.

No server, neural network, optimizer or prerecorded aircraft state is used.
The external actor supplies the original five categorical actuator indices.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

import a4_math
import crash_math
import geometry_context
import observation_math
import open_geometry
import stream_math
from guidance import A4ClosedLoopGuidanceConfig, bind_a4_guidance_to_stream_state
from lifted_guidance import LiftedPitchGuidance
from progress import Progress

ACTION_DIMS = np.asarray([31,81,41,41,5],np.int32)
CONTROL_DT = np.float32(.2)
SUBSTEP_DT = np.float32(.02)
TRIM_ALPHA = np.deg2rad(np.float32(.6))
TRIM_ELEVATOR = np.float32(-.010555555112659931)
CHECKPOINT = '9a45713d3f392f7762758b27535e8057f271035880f00bbd99197efcde878bc0'


class Record:
    def __init__(self, **values):
        self.__dict__.update(values)
    def replace(self, **values):
        return type(self)(**dict(vars(self), **values))


class Plane(Record):
    @property
    def is_alive(self):
        return self.status == 0
    @property
    def is_locked(self):
        return self.status == 1
    @property
    def is_crashed(self):
        return self.status == 2


def batched(record):
    return type(record)(**{k:np.asarray([v],dtype=np.asarray(v).dtype) for k,v in vars(record).items()})


def trim_plane(position, heading):
    p = {k:np.float32(0) for k in (
        'north','east','altitude','roll','pitch','yaw','vel_x','vel_y','vel_z',
        'vt','alpha','beta','P','Q','R','q0','q1','q2','q3','T','el','ail','rud',
        'lef','sb','ax','ay','az','aero_scale','mass_scale','inertia_scale',
        'wind_body_u','wind_body_v','wind_body_w','blood')}
    q = observation_math._quat_from_euler_nb(np.float32(0),TRIM_ALPHA,np.float32(heading))
    q = q*np.asarray([1,-1,-1,-1],np.float32)
    p.update(north=np.float32(position[0]),east=np.float32(position[1]),altitude=np.float32(position[2]),
        roll=np.float32(0),pitch=TRIM_ALPHA,yaw=np.float32(heading),
        vel_x=np.float32(250)*np.cos(np.float32(heading)),vel_y=np.float32(250)*np.sin(np.float32(heading)),
        vt=np.float32(250),alpha=TRIM_ALPHA,q0=q[0],q1=q[1],q2=q[2],q3=q[3],
        T=np.float32(2836.600830078125),el=TRIM_ELEVATOR,
        aero_scale=np.float32(1),mass_scale=np.float32(1),inertia_scale=np.float32(1),
        status=np.int32(0),blood=np.float32(100))
    return Plane(**p)


def decode_action(action):
    a = np.asarray(action)
    if a.shape != (5,) or not np.issubdtype(a.dtype,np.integer) or np.any(a < 0) or np.any(a >= ACTION_DIMS):
        raise ValueError('五通道动作必须是原 baseline 合法整数索引')
    a = a.astype(np.float32)
    return np.asarray([a[0]/30.,a[1]*2./80.-1.,a[2]*2./40.-1.,a[3]*2./40.-1.,a[4]/4.],np.float32)


class Flight:
    """One ordered route and one continuous physical A4 episode.

    observe() -> float32[21]; context() -> float32[36]. Both refer to the
    current state and must be read before advance(action_indices[5]).
    """
    def __init__(self, request):
        self.geometry = open_geometry.prepare(request)
        self.stream, load, gamma = open_geometry.synthesize(self.geometry)
        self.compilation = open_geometry.diagnose(self.geometry,self.stream,load,gamma)
        self.reference = self.stream.positions_neu_m
        self.reference_phase = self.stream.phase_s_m
        self.targets = np.asarray(self.stream.targets[None],np.float32)
        self.lengths = np.asarray([len(self.stream.targets)],np.int32)
        self.plane = trim_plane(self.reference[0],np.float32(np.radians(self.geometry['heading'])))
        self.control = Record(throttle=np.float32(.34),elevator=TRIM_ELEVATOR/np.float32(.35),
            aileron=np.float32(0),rudder=np.float32(0),speed_brake=np.float32(0))
        self.controls = 0
        self.done = False
        self.reason = None
        self._observation = None
        self._context = None
        self.progress = Progress(self.reference,self.reference_phase)
        self.guidance = LiftedPitchGuidance(self.reference,self.reference_phase,self.stream.targets,A4ClosedLoopGuidanceConfig())
        initial = self.hprv()
        self.guidance.reset(initial)
        self.bounds = stream_math.StreamBounds(heading_slew_rad_s=np.deg2rad(np.float32(30)),
            pitch_slew_rad_s=np.deg2rad(np.float32(16)),roll_slew_rad_s=np.deg2rad(np.float32(50)),
            speed_slew_m_s2=np.float32(15),pitch_limit_rad=np.float32(4*np.pi),roll_limit_rad=np.float32(np.pi),
            min_speed_m_s=np.float32(170),max_speed_m_s=np.float32(300))
        v = lambda x:np.asarray([x],np.float32)
        cursor = stream_math.StreamCursor(stream_id=np.asarray([0],np.int32),phase=v(0),phase_step=v(1),
            mirror_sign=v(1),amplitude_scale=v(1),time_scale=v(1),heading_rotation=v(0),
            anchor_phase=v(0),anchor_target=initial[None],current_target=initial[None],
            certified_fidelity=np.asarray([True],np.bool_))
        self.state = Record(plane_state=batched(self.plane),control_state=batched(self.control),
            target_heading=v(initial[0]),target_pitch=v(initial[1]),target_roll=v(initial[2]),target_vt=v(initial[3]),
            stream_cursor=cursor,stream_current_target=initial[None],stream_next_target=initial[None],
            stream_target_rates=np.zeros((1,4),np.float32),stream_complete=np.asarray([False]),
            stream_completion_age=np.asarray([0],np.int32),stream_phase_advance=v(0))
        command = self.guidance.step(self.position(),float(self.plane.vt))
        self.state = bind_a4_guidance_to_stream_state(self.state,command,control_dt_s=float(CONTROL_DT))

    def position(self):
        return np.asarray([self.plane.north,self.plane.east,self.plane.altitude],np.float32)

    def hprv(self):
        return np.asarray([self.plane.yaw,self.plane.pitch,self.plane.roll,self.plane.vt],np.float32)

    def observe(self):
        if self._observation is None:
            view = self.state
            target = np.asarray(view.stream_next_target,np.float32)
            view = view.replace(target_heading=target[:,0],target_pitch=target[:,1],target_roll=target[:,2],target_vt=target[:,3])
            obs = observation_math.base_observation(SimpleNamespace(num_agents=1,agents=('ally_0',)),view,None)['ally_0']
            code, valid = stream_math.future_twist_preview_code(self.state.stream_cursor,self.targets,self.lengths,
                self.bounds,float(CONTROL_DT),horizon_steps=4,rate_limit_deg_s=55.,code_scale=.5,zero_ablation=False)
            if not np.asarray(valid).all():
                raise ValueError('原始 future-twist 预览数值校验失败')
            obs = np.array(obs,dtype=np.float32,copy=True)
            obs[6:9] = code[0]
            self._observation = obs
        return self._observation.copy()

    def context(self):
        if self._context is None:
            self._context = geometry_context.encode(self.reference,self.reference_phase,
                cursor=float(self.state.stream_cursor.phase[0]),position=self.position(),
                yaw=float(self.plane.yaw),speed=float(self.plane.vt))
        return self._context.copy()

    def advance(self, action_indices):
        if self.done:
            raise RuntimeError('此真实飞行已经结束；请新建 Flight')
        normalized = decode_action(action_indices)
        self.control = Record(**dict(zip(('throttle','elevator','aileron','rudder','speed_brake'),normalized)))
        previous = self.state
        # Original AeroPlanaxEnv.step repeats action decoding and status checks
        # on each physical substep, and freezes dynamics after a crash status.
        for _ in range(10):
            self.plane = a4_math.update(self.plane,self.control,SUBSTEP_DT)
            if bool(crash_math.check_crashed(batched(self.plane),0)) and bool(self.plane.is_alive | self.plane.is_locked):
                self.plane = self.plane.replace(status=np.int32(2))
        self.controls += 1
        self.state = self.state.replace(plane_state=batched(self.plane),control_state=batched(self.control))
        self.progress.update(self.position(),self.controls*.2)
        finite = all(np.isfinite(np.asarray(v)).all() for v in vars(self.plane).values())
        physical_done = bool(self.plane.is_crashed) or bool(np.any(np.abs([self.plane.ax,self.plane.ay,self.plane.az]) > 10)) or not finite
        if physical_done:
            self.reason = 'physical_termination' if finite else 'nonfinite_physics'
        elif self.progress.reason:
            self.reason = self.progress.reason
        # The old env advances its cursor before host-side geometric rebinding.
        # Rebinding uses the previous physical step's guidance state explicitly.
        if not self.reason:
            cursor, targets = stream_math.advance_stream_cursor(self.state.stream_cursor,self.targets,self.lengths,self.bounds,float(CONTROL_DT))
            self.state = self.state.replace(stream_cursor=cursor,stream_next_target=targets.next,
                stream_current_target=targets.current,stream_target_rates=targets.rates)
            command = self.guidance.step(self.position(),float(self.plane.vt))
            self.state = bind_a4_guidance_to_stream_state(self.state,command,control_dt_s=float(CONTROL_DT),previous_guidance_state=previous)
        self.done = self.reason is not None
        self._observation = None
        self._context = None
        return self.sample()

    def cancel(self):
        self.done = True
        self.reason = 'cancelled'
        return self.sample()

    def sample(self):
        return dict(time=self.controls*.2,position=self.position().tolist(),hprv=self.hprv().tolist(),
            alpha=float(np.rad2deg(self.plane.alpha)),beta=float(np.rad2deg(self.plane.beta)),
            g=float(np.linalg.norm(np.asarray([self.plane.ax,self.plane.ay,self.plane.az],np.float32))),
            progress=float(self.progress.fraction),controls=self.controls,done=self.done,reason=self.reason,
            cursor=float(self.state.stream_cursor.phase[0]),target=self.state.stream_next_target[0].tolist(),
            synthetic_flight_samples=False)

    def identity(self):
        return dict(checkpoint=CHECKPOINT,physics='original A4 NumPy browser execution',
            control_dt_s=.2,dynamics_substeps=10,dynamics_dt_s=.02,observation_dim=21,context_dim=36,
            action_dims=ACTION_DIMS.tolist(),teacher_calls=0,optimizer_updates=0,checkpoint_switches=0,
            server_calls=0,formal_research_run=False,bitwise_equivalence_claim=False)
