from typing import NamedTuple


class AircraftSpec(NamedTuple):
    name: str
    mass_slug: float
    wing_area_ft2: float
    wingspan_ft: float
    chord_ft: float
    jx_slug_ft2: float
    jy_slug_ft2: float
    jz_slug_ft2: float
    jxz_slug_ft2: float
    max_thrust_lbf: float
    elevator_min_rad: float
    elevator_max_rad: float
    aileron_min_rad: float
    aileron_max_rad: float
    rudder_min_rad: float
    rudder_max_rad: float
    speedbrake_min: float
    speedbrake_max: float
    throttle_tau_s: float
    surface_tau_s: float
    speedbrake_tau_s: float


F16_SPEC = AircraftSpec(
    name="F16_like_existing",
    mass_slug=636.94,
    wing_area_ft2=300.0,
    wingspan_ft=30.0,
    chord_ft=11.32,
    jx_slug_ft2=9496.0,
    jy_slug_ft2=55814.0,
    jz_slug_ft2=63100.0,
    jxz_slug_ft2=982.0,
    max_thrust_lbf=76300.0 * 0.225 / 0.3048,
    elevator_min_rad=-0.7853981633974483,
    elevator_max_rad=0.7853981633974483,
    aileron_min_rad=-0.7853981633974483,
    aileron_max_rad=0.7853981633974483,
    rudder_min_rad=-0.7853981633974483,
    rudder_max_rad=0.7853981633974483,
    speedbrake_min=0.0,
    speedbrake_max=1.0471975511965976,
    throttle_tau_s=0.19,
    surface_tau_s=0.19,
    speedbrake_tau_s=0.19,
)


A4_SPEC = AircraftSpec(
    name="A4_JSBSim_minimal",
    mass_slug=13250.0 / 32.174,
    wing_area_ft2=260.0,
    wingspan_ft=27.50,
    chord_ft=9.45,
    jx_slug_ft2=5778.0,
    jy_slug_ft2=25900.0,
    jz_slug_ft2=20082.0,
    jxz_slug_ft2=0.0,
    max_thrust_lbf=11200.0,
    elevator_min_rad=-0.35,
    elevator_max_rad=0.30,
    aileron_min_rad=-0.35,
    aileron_max_rad=0.35,
    rudder_min_rad=-0.35,
    rudder_max_rad=0.35,
    speedbrake_min=0.0,
    speedbrake_max=1.0,
    throttle_tau_s=0.60,
    surface_tau_s=0.25,
    speedbrake_tau_s=1.00,
)
