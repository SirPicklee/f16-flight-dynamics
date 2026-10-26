import numpy as np
import pytest

from f16.dynamics import ALPHA, BETA, P, R
from f16.fcs import FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_ari, design_roll_command
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


def roll_pilot(t):
    p = np.zeros(4)
    p[2] = 60.0 if 0.5 <= t < 1.5 else 0.0
    return p


def max_sideslip(high_alpha, stability_axis, ari):
    r, a, b, g = high_alpha
    roll = design_roll_command(a, b)
    k_ari = design_ari(a, b, r.x[ALPHA], roll[1]) if ari else 0.0
    fcs = FlightControlSystem(r.x, r.u, g, roll_command=roll,
                              stability_axis=stability_axis, ari_gain=k_ari)
    _, z = fcs.simulate(roll_pilot, 3.0)
    return np.abs(np.degrees(z[:, BETA])).max()


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
    # Measured: about 40 % less sideslip with the yaw damper alone (7.4 -> 4.5 deg)
    body = max_sideslip(high_alpha, stability_axis=False, ari=False)
    stab = max_sideslip(high_alpha, stability_axis=True, ari=False)
    assert stab < 0.75 * body


def test_ari_balances_yaw_in_a_steady_stability_axis_roll(high_alpha):
    r, a, b, _ = high_alpha
    _, kff = design_roll_command(a, b)
    k_ari = design_ari(a, b, r.x[ALPHA], kff)
    p = np.radians(1.0)                                  # 1 deg/s roll rate
    yaw_accel = (a[R, P] * p + a[R, R] * p * np.tan(r.x[ALPHA])
                 + b[R, 2] * kff + b[R, 3] * k_ari)
    assert yaw_accel == pytest.approx(0.0, abs=1e-12)


def test_ari_reduces_sideslip_further(high_alpha):
    stab = max_sideslip(high_alpha, stability_axis=True, ari=False)
    stab_ari = max_sideslip(high_alpha, stability_axis=True, ari=True)
    assert stab_ari < stab