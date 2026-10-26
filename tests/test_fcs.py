import numpy as np
import pytest

from f16.dynamics import ALPHA, Q, R
from f16.fcs import (DA, DE, DR, ELEVATOR, XD, XF, XW, FlightControlSystem,
                     lateral_loop, loop_margins, pitch_loop)
from f16.linearize import linearize, modes
from f16.monte_carlo import LeadFilter
from f16.sas import design_pitch_sas, design_yaw_damper
from f16.simulate import doublet
from f16.trim import trim_level


@pytest.fixture(scope="module")
def setup():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    g.k_r = design_yaw_damper(a, b, zeta=0.5)
    return r, a, b, g


def test_actuator_respects_rate_and_position_limits():
    assert ELEVATOR.rate(0.0, 100.0) == pytest.approx(ELEVATOR.rate_limit)
    assert ELEVATOR.rate(0.0, -100.0) == pytest.approx(-ELEVATOR.rate_limit)
    assert ELEVATOR.rate(ELEVATOR.position_limit, 100.0) == pytest.approx(0.0)


def test_fcs_starts_in_equilibrium(setup):
    r, _, _, g = setup
    fcs = FlightControlSystem(r.x, r.u, g)
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 7, 13, 14, 15, 16]])) < 1e-9


def test_pitch_loop_meets_classic_margin_requirements(setup):
    # Common requirement for flight control loops: 6 dB gain, 45 deg phase
    _, a, b, g = setup
    m = loop_margins(*pitch_loop(a, b, g))
    assert m.phase_margin_deg > 45.0
    assert m.gain_reduction_db < -6.0
    assert m.delay_margin > 0.1


def test_pitch_loop_stable_with_actuator(setup):
    _, a, b, g = setup
    aa, ba, k = pitch_loop(a, b, g)
    assert all(md.eigenvalue.real < 0 for md in modes(aa + ba @ k))


def test_washout_keeps_dutch_roll_damped_and_spiral_free(setup):
    _, a, b, g = setup
    aa, ba, k = lateral_loop(a, b, g)
    ms = modes(aa + ba @ k)
    (dutch_roll,) = [m for m in ms if m.is_oscillatory]
    spiral = min((m for m in ms if not m.is_oscillatory), key=lambda m: m.wn)
    assert dutch_roll.zeta > 0.4
    assert spiral.time_constant > 20.0  # turns are not opposed by the damper


def test_washout_gives_no_rudder_in_a_steady_turn(setup):
    r, _, _, g = setup
    fcs = FlightControlSystem(r.x, r.u, g)
    z = fcs.initial_state()
    z[R] = 0.1
    z[XW] = 0.1  # filter has settled on the constant yaw rate
    cmd = fcs.commands(z, np.zeros(4))
    assert cmd[2] == pytest.approx(r.u[3])


def test_large_input_hits_elevator_rate_limit(setup):
    r, _, _, g = setup
    fcs = FlightControlSystem(r.x, r.u, g)

    def pilot(t):
        p = np.zeros(4)
        p[1] = doublet(t, 1.0, 0.3, 10.0)
        return p

    t, z = fcs.simulate(pilot, 4.0)
    rate = np.diff(z[:, DE]) / np.diff(t)
    assert np.max(np.abs(rate)) == pytest.approx(ELEVATOR.rate_limit, rel=0.01)


def test_nonlinear_fcs_returns_to_trim(setup):
    r, _, _, g = setup
    fcs = FlightControlSystem(r.x, r.u, g)

    def pilot(t):
        p = np.zeros(4)
        p[1] = doublet(t, 1.0, 1.0, 1.0)
        return p

    _, z = fcs.simulate(pilot, 15.0)
    assert np.degrees(abs(z[-1, ALPHA] - r.x[ALPHA])) < 0.01


def test_margins_are_infinite_without_crossover():
    # Stable first-order plant with a very small feedback gain: |L| < 1 everywhere
    aa = np.array([[-1.0]])
    ba = np.array([[1.0]])
    k = np.array([[-0.01]])
    m = loop_margins(aa, ba, k)
    assert m.phase_margin_deg == np.inf
    assert m.delay_margin == np.inf
    assert m.gain_reduction_db == -np.inf


def test_lead_and_delay_start_in_equilibrium(setup):
    r, _, _, g = setup
    fcs = FlightControlSystem(r.x, r.u, g, lead=LeadFilter(6.0, 2.5), delay=0.045)
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 7, DE, DA, DR, XW, XF, XD]])) < 1e-9


def test_neutral_lead_matches_plain_fcs(setup):
    r, _, _, g = setup
    plain = FlightControlSystem(r.x, r.u, g)
    neutral = FlightControlSystem(r.x, r.u, g, lead=LeadFilter(6.0, 1.0))
    z = plain.initial_state()
    z[ALPHA] += np.radians(1.0)
    z[Q] += np.radians(2.0)
    z[XF] = 0.3
    assert neutral.commands(z, np.zeros(4)) == pytest.approx(plain.commands(z, np.zeros(4)))


def test_lead_and_delay_pass_steady_signals_unchanged(setup):
    # Once the filter and the delay have settled, a constant SAS signal is unchanged
    r, _, _, g = setup
    lead = LeadFilter(6.0, 2.5)
    plain = FlightControlSystem(r.x, r.u, g)
    fcs = FlightControlSystem(r.x, r.u, g, lead=lead, delay=0.045)
    z = plain.initial_state()
    z[ALPHA] += np.radians(1.0)
    e = g.k_alpha * 1.0             # SAS signal for 1 deg of alpha
    z[XF] = e / lead.pole           # filter settled
    z[XD] = e                       # delay settled
    assert fcs.commands(z, np.zeros(4))[0] == pytest.approx(plain.commands(z, np.zeros(4))[0])