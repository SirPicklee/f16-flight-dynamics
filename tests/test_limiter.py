import numpy as np
import pytest

from f16.dynamics import ALPHA
from f16.fcs import XI, EnvelopeLimiter, FlightControlSystem
from f16.gcommand import design_g_command
from f16.linearize import linearize
from f16.trim import trim_level


@pytest.fixture(scope="module")
def setup():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g, ki = design_g_command(a, b, r.x, r.u)
    return r, g, ki


def make_fcs(setup, limiter=EnvelopeLimiter(), anti_windup=True):
    r, g, ki = setup
    return FlightControlSystem(r.x, r.u, g, g_command=(0.0, ki),
                               limiter=limiter, anti_windup=anti_windup)


def pilot(nz):
    return np.array([0.0, nz, 0.0, 0.0])


def test_without_limiter_the_command_passes_unchanged(setup):
    fcs = make_fcs(setup, limiter=None)
    assert fcs.nz_command(fcs.initial_state(), pilot(12.0)) == 12.0


def test_g_command_is_clipped_at_the_load_factor_limit(setup):
    fcs = make_fcs(setup)
    r = fcs.nz_command(fcs.initial_state(), pilot(20.0))
    assert r == pytest.approx(9.0 - fcs.nz0)


def test_alpha_limiter_reduces_the_command_above_threshold(setup):
    fcs = make_fcs(setup)
    lim = fcs.limiter
    z = fcs.initial_state()
    z[ALPHA] = np.radians(25.0)
    expected = 2.0 - lim.alpha_gain * (25.0 - lim.alpha_start)
    assert fcs.nz_command(z, pilot(2.0)) == pytest.approx(expected)


def test_limiter_starts_in_equilibrium(setup):
    fcs = make_fcs(setup)
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 7, 13, XI]])) < 1e-9


@pytest.mark.parametrize("anti_windup, frozen", [(True, True), (False, False)])
def test_integrator_freezes_while_elevator_is_saturated(setup, anti_windup, frozen):
    fcs = make_fcs(setup, anti_windup=anti_windup)
    z = fcs.initial_state()
    z[XI] = 100.0                            # integrator far beyond the elevator limit
    p = pilot(1.0)                           # positive error: wants even more nose up
    assert fcs.commands(z, p)[0] < -fcs.elevator.position_limit
    zd = fcs.derivatives(0.0, z, lambda t: p)
    assert bool(zd[XI] == 0.0) == frozen


def full_stick(t):
    return pilot(8.0 if t >= 0.5 else 0.0)


@pytest.mark.parametrize("limiter, bound", [(None, 30.0), (EnvelopeLimiter(), 25.5)])
def test_full_aft_stick_alpha(setup, limiter, bound):
    fcs = make_fcs(setup, limiter=limiter)
    _, z = fcs.simulate(full_stick, 3.0)
    max_alpha = np.degrees(z[:, ALPHA]).max()
    if limiter is None:
        assert max_alpha > bound             # unprotected: alpha runs away
    else:
        assert max_alpha < bound             # protected: held near the limit