import numpy as np
import pytest

from f16.dynamics import ALPHA, BETA, VT
from f16.linearize import LAT_STATES, linearize, modes, submatrix
from f16.simulate import doublet, simulate_linear, simulate_nonlinear
from f16.trim import trim_level

XCG = 0.30  # open-loop stable configuration


@pytest.fixture(scope="module")
def trimmed():
    r = trim_level(502.0, xcg=XCG)
    a, b = linearize(r.x, r.u, xcg=XCG)
    return r, a, b


def test_doublet_shape():
    assert doublet(0.5, 1.0, 1.0, 2.0) == 0.0
    assert doublet(1.5, 1.0, 1.0, 2.0) == 2.0
    assert doublet(2.5, 1.0, 1.0, 2.0) == -2.0
    assert doublet(3.5, 1.0, 1.0, 2.0) == 0.0


def test_aircraft_stays_trimmed_without_input(trimmed):
    r, _, _ = trimmed
    _, x = simulate_nonlinear(r.x, lambda t: r.u, 10.0, xcg=XCG)
    assert np.max(np.abs(x[:, VT] - r.x[VT])) < 1e-3
    assert np.max(np.abs(x[:, ALPHA] - r.x[ALPHA])) < 1e-6


def test_linear_model_matches_nonlinear_for_small_input(trimmed):
    r, a, b = trimmed

    def control(t):
        u = r.u.copy()
        u[1] += doublet(t, 1.0, 1.0, 1.0)
        return u

    _, xn = simulate_nonlinear(r.x, control, 10.0, xcg=XCG)
    _, xl = simulate_linear(a, b, r.x, r.u, control, 10.0)
    peak = np.max(np.abs(xn[:, ALPHA] - r.x[ALPHA]))
    error = np.max(np.abs(xn[:, ALPHA] - xl[:, ALPHA]))
    assert error < 0.02 * peak


def test_simulated_dutch_roll_period_matches_eigenvalue(trimmed):
    r, a, _ = trimmed
    (dutch_roll,) = [m for m in modes(submatrix(a, LAT_STATES)) if m.is_oscillatory]

    def control(t):
        u = r.u.copy()
        u[3] += doublet(t, 1.0, 1.0, 2.0)
        return u

    t, x = simulate_nonlinear(r.x, control, 15.0, xcg=XCG)
    mask = t > 4.0                       # free response after the input
    beta, tt = x[mask, BETA], t[mask]
    crossings = tt[1:][np.diff(np.sign(beta)) != 0]
    measured_period = 2.0 * np.mean(np.diff(crossings))
    assert measured_period == pytest.approx(dutch_roll.period, rel=0.03)