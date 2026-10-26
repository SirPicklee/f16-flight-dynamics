import numpy as np
import pytest

from f16.dynamics import ALPHA, BETA, P, R
from f16.fcs import FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_roll_command
from f16.sas import design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

VT0 = 260.0


@pytest.fixture(scope="module")
def high_alpha():
    r = trim_level(VT0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b)
    g.k_r = design_yaw_damper(a, b)
    return r, a, b, g


def test_trim_is_at_high_alpha(high_alpha):
    r, _, _, _ = high_alpha
    assert r.success
    assert r.alpha_deg > 10.0


def test_stability_axis_yaw_signal(high_alpha):
    r, _, _, g = high_alpha
    body = FlightControlSystem(r.x, r.u, g)
    stab = FlightControlSystem(r.x, r.u, g, stability_axis=True)
    z = body.initial_state()
    z[P], z[R] = 0.5, 0.1
    assert body.yaw_rate_signal(z) == pytest.approx(0.1)
    assert stab.yaw_rate_signal(z) == pytest.approx(0.1 - 0.5 * np.tan(z[ALPHA]))


def test_stability_axis_starts_in_equilibrium(high_alpha):
    r, a, b, g = high_alpha
    fcs = FlightControlSystem(r.x, r.u, g, roll_command=design_roll_command(a, b),
                              stability_axis=True)
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 2, 6, 7, 8, 13, 14, 15, 16]])) < 1e-9


def test_stability_axis_roll_reduces_sideslip(high_alpha):
    r, a, b, g = high_alpha
    roll = design_roll_command(a, b)

    def pilot(t):
        p = np.zeros(4)
        p[2] = 60.0 if 0.5 <= t < 1.5 else 0.0
        return p

    max_beta = {}
    for stability_axis in (False, True):
        fcs = FlightControlSystem(r.x, r.u, g, roll_command=roll,
                                  stability_axis=stability_axis)
        _, z = fcs.simulate(pilot, 3.0)
        max_beta[stability_axis] = np.abs(np.degrees(z[:, BETA])).max()
    # Measured: about 40 % less sideslip with the yaw damper alone (7.4 -> 4.5 deg)
    assert max_beta[True] < 0.75 * max_beta[False]