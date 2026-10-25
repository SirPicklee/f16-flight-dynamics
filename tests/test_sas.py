import numpy as np
import pytest

from f16.dynamics import ALPHA
from f16.linearize import LAT_STATES, LONG_STATES, linearize, modes, submatrix
from f16.sas import (EL, ELEVATOR_LIMIT, closed_loop_matrix, design_pitch_sas,
                     design_yaw_damper, sas_control)
from f16.simulate import doublet, simulate_closed_loop
from f16.trim import trim_level


@pytest.fixture(scope="module")
def design():
    r = trim_level(502.0)  # nominal, unstable CG
    a, b = linearize(r.x, r.u)
    gains = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    gains.k_r = design_yaw_damper(a, b, zeta=0.5)
    return r, a, b, gains


def oscillatory(a, states):
    return [m for m in modes(submatrix(a, states)) if m.is_oscillatory]


def test_short_period_placed_where_requested(design):
    _, a, b, g = design
    short_period = max(oscillatory(closed_loop_matrix(a, b, g), LONG_STATES),
                       key=lambda m: m.wn)
    assert short_period.wn == pytest.approx(3.0, abs=1e-6)
    assert short_period.zeta == pytest.approx(0.7, abs=1e-6)


def test_yaw_damper_sets_dutch_roll_damping(design):
    _, a, b, g = design
    (dutch_roll,) = oscillatory(closed_loop_matrix(a, b, g), LAT_STATES)
    assert dutch_roll.zeta == pytest.approx(0.5, abs=1e-6)


def test_sas_stabilizes_nominal_cg(design):
    _, a, b, g = design
    acl = closed_loop_matrix(a, b, g)
    for states in (LONG_STATES, LAT_STATES):
        assert all(m.eigenvalue.real < 0 for m in modes(submatrix(acl, states)))


@pytest.mark.parametrize("vt, xcg", [(502.0, 0.30), (502.0, 0.38),
                                     (350.0, 0.35), (700.0, 0.35)])
def test_fixed_gains_stay_stable_off_design(design, vt, xcg):
    _, _, _, g = design
    r = trim_level(vt, xcg=xcg)
    a, b = linearize(r.x, r.u, xcg=xcg)
    acl = closed_loop_matrix(a, b, g)
    assert all(m.eigenvalue.real < 0 for m in modes(submatrix(acl, LONG_STATES)))


def test_nonlinear_aircraft_returns_to_trim_with_sas(design):
    r, _, _, g = design

    def controller(t, x):
        pilot = np.zeros(4)
        pilot[EL] = doublet(t, 1.0, 1.0, 1.0)
        return sas_control(x, r.x, r.u, g, pilot)

    _, x = simulate_closed_loop(r.x, controller, 15.0)
    assert np.degrees(abs(x[-1, ALPHA] - r.x[ALPHA])) < 0.01


def test_surface_commands_are_limited(design):
    r, _, _, g = design
    x = r.x.copy()
    x[ALPHA] += np.radians(60.0)  # absurd disturbance
    u = sas_control(x, r.x, r.u, g)
    assert abs(u[EL]) <= ELEVATOR_LIMIT