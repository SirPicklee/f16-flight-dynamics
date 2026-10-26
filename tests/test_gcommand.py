import numpy as np
import pytest
from scipy.signal import StateSpace, step

from f16.gcommand import (design_g_command, nz_command_loop, nz_output_row,
                          step_metrics)
from f16.linearize import linearize
from f16.sas import design_pitch_sas
from f16.trim import trim_level


@pytest.fixture(scope="module")
def nominal():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    return r, a, b, g


def test_nz_grows_with_alpha_and_with_trailing_edge_down_elevator(nominal):
    r, _, _, _ = nominal
    row = nz_output_row(r.x, r.u)
    assert row[1] > 0.0   # more angle of attack, more lift
    assert row[4] > 0.0   # tail lift: positive elevator pushes the tail up


def test_integral_action_gives_unit_steady_state_gain(nominal):
    r, a, b, g = nominal
    m, bm, cm = nz_command_loop(a, b, r.x, r.u, g, kp=1.0, ki=5.0)
    metrics = step_metrics(m, bm, cm)
    assert metrics.stable
    assert metrics.dc_gain == pytest.approx(1.0, abs=1e-6)


def test_without_integral_the_sas_leaves_a_steady_error(nominal):
    r, a, b, g = nominal
    m, bm, cm = nz_command_loop(a, b, r.x, r.u, g, kp=1.0, ki=0.0)
    m3, bm3, cm3 = m[:-1, :-1], bm[:-1], cm[:, :-1]   # drop the unused integrator
    _, y = step(StateSpace(m3, bm3, cm3, np.zeros((1, 1))), T=np.linspace(0, 3, 301))
    assert abs(np.asarray(y).ravel()[-1] - 1.0) > 0.2


def test_full_model_has_only_a_slow_path_instability(nominal):
    # A sustained g command curves the flight path without end: on the full
    # longitudinal model only a very slow speed/path mode is unstable
    r, a, b, g = nominal
    m, _, _ = nz_command_loop(a, b, r.x, r.u, g, kp=1.0, ki=5.0, short_period=False)
    roots = np.linalg.eigvals(m)
    unstable = roots[roots.real > 0.0]
    assert len(unstable) == 1
    assert unstable[0].real < 0.01


def test_pole_sum_depends_on_kp_but_not_on_ki(nominal):
    # The trace of the loop matrix is the sum of its eigenvalues
    r, a, b, g = nominal

    def pole_sum(kp, ki):
        m, _, _ = nz_command_loop(a, b, r.x, r.u, g, kp, ki)
        return float(np.sum(np.linalg.eigvals(m).real))

    assert pole_sum(1.0, 2.0) == pytest.approx(pole_sum(1.0, 9.0), abs=1e-9)
    assert pole_sum(2.0, 5.0) > pole_sum(0.0, 5.0) + 0.5


def test_proportional_nz_feedback_slows_the_slowest_pole(nominal):
    r, a, b, g = nominal
    slowest = []
    for kp in (1.0, 4.0):
        m, _, _ = nz_command_loop(a, b, r.x, r.u, g, kp, 2.0)
        slowest.append(np.max(np.linalg.eigvals(m).real))
    assert slowest[1] > slowest[0]


def test_joint_design_places_all_slow_poles(nominal):
    r, a, b, _ = nominal
    g, ki = design_g_command(a, b, r.x, r.u, wn=3.0, zeta=0.7, integral_pole=3.0)
    assert g.k_alpha > 0.0 and g.k_q > 0.0 and ki > 0.0
    m, _, _ = nz_command_loop(a, b, r.x, r.u, g, 0.0, ki)
    roots = np.linalg.eigvals(m)
    assert np.all(roots.real < 0.0)
    assert np.min(np.abs(roots - complex(-2.1, 2.1424))) < 1e-3
    assert np.min(np.abs(roots + 3.0)) < 1e-3