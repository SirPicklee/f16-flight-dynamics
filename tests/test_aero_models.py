import numpy as np
import pytest

from f16.dynamics import aero_coefficients, f16_derivatives
from f16.envelope import survey_point
from f16.linearize import LONG_STATES, linearize, submatrix
from f16.sas import closed_loop_matrix
from f16.schedule import build_schedule
from f16.trim import trim_level


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        f16_derivatives(np.r_[502.0, np.zeros(12)], np.zeros(4), model="spline")


def test_models_agree_on_nominal_trim():
    tables = trim_level(502.0, model="tables")
    poly = trim_level(502.0, model="morelli")
    assert poly.success
    assert poly.throttle == pytest.approx(tables.throttle, abs=0.03)
    assert poly.alpha_deg == pytest.approx(tables.alpha_deg, abs=1.0)


def test_models_agree_on_nominal_pitch_divergence():
    roots = []
    for model in ("tables", "morelli"):
        r = trim_level(502.0, model=model)
        a, _ = linearize(r.x, r.u, model=model)
        roots.append(np.max(np.linalg.eigvals(submatrix(a, LONG_STATES)).real))
    assert roots[1] == pytest.approx(roots[0], rel=0.1)


def test_polynomial_pitching_moment_is_linear_in_alpha_at_zero_elevator():
    alphas = np.linspace(-10.0, 45.0, 12)
    cm = np.array([aero_coefficients(500.0, a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.35, "morelli")[4] for a in alphas])
    assert np.max(np.abs(np.diff(cm, 2))) < 1e-12


def test_polynomial_model_removes_stability_jumps():
    speeds = np.arange(300.0, 701.0, 10.0)
    jumps = {}
    for model in ("tables", "morelli"):
        roots = [survey_point(v, 0.0, model=model).max_real_long for v in speeds]
        jumps[model] = np.max(np.abs(np.diff(roots)))
    assert jumps["tables"] > 0.1
    assert jumps["morelli"] < 0.01


@pytest.fixture(scope="module")
def schedule():
    return build_schedule()


@pytest.mark.parametrize("vt, alt", [(375.0, 7500.0), (625.0, 22500.0), (725.0, 37500.0)])
def test_schedule_designed_on_tables_stabilizes_polynomial_model(schedule, vt, alt):
    r = trim_level(vt, alt=alt, model="morelli")
    a, b = linearize(r.x, r.u, model="morelli")
    acl = closed_loop_matrix(a, b, schedule.gains(vt, alt))
    assert np.max(np.linalg.eigvals(submatrix(acl, LONG_STATES)).real) < 0.0