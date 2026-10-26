import numpy as np
import pytest

from f16.dynamics import P, PHI
from f16.fcs import FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_roll_command, roll_mode
from f16.sas import design_pitch_sas, design_yaw_damper
from f16.trim import trim_level


@pytest.fixture(scope="module")
def setup():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b)
    g.k_r = design_yaw_damper(a, b)
    return r, a, b, g


def test_roll_mode_signs(setup):
    _, a, b, _ = setup
    m = roll_mode(a, b)
    assert m.lp < 0.0      # roll damping
    assert m.lda < 0.0     # positive aileron rolls left on this model


def test_design_places_roll_pole_with_unit_steady_state_gain(setup):
    _, a, b, _ = setup
    m = roll_mode(a, b)
    kp, kff = design_roll_command(a, b, bandwidth=6.0)
    assert m.lp - m.lda * kp == pytest.approx(-6.0)
    steady_gain = -m.lda * (kff + kp) / (m.lp - m.lda * kp)
    assert steady_gain == pytest.approx(1.0)


def test_roll_command_starts_in_equilibrium(setup):
    r, a, b, g = setup
    fcs = FlightControlSystem(r.x, r.u, g, roll_command=design_roll_command(a, b))
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 2, 6, 7, 8, 13, 14, 15]])) < 1e-9


def test_nonlinear_aircraft_follows_a_roll_rate_command(setup):
    r, a, b, g = setup
    fcs = FlightControlSystem(r.x, r.u, g, roll_command=design_roll_command(a, b))

    def pilot(t):
        p = np.zeros(4)
        p[2] = 30.0 if t >= 0.5 else 0.0    # 30 deg/s roll rate command
        return p

    t, z = fcs.simulate(pilot, 1.5)
    p_end = np.degrees(z[-1, P])
    assert p_end == pytest.approx(30.0, rel=0.1)
    lag = 1.0 / 6.0 + 1.0 / 20.2          # roll pole and aileron actuator time constants
    expected_bank = 30.0 * (1.0 - lag)    # a ramp through a lag trails by the lag time
    assert np.degrees(z[-1, PHI]) == pytest.approx(expected_bank, abs=2.0)