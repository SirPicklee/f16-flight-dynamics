import pytest

from f16.linearize import LONG_STATES, linearize, modes, submatrix
from f16.sas import closed_loop_matrix
from f16.schedule import build_cg_schedule
from f16.trim import trim_level

WN = 2.5
ZETA = 0.7


@pytest.fixture(scope="module")
def cg_schedule():
    return build_cg_schedule(wn=WN)


def short_period(vt, alt, xcg, gains):
    r = trim_level(vt, alt=alt, xcg=xcg)
    a, b = linearize(r.x, r.u, xcg=xcg)
    acl = closed_loop_matrix(a, b, gains)
    return max((m for m in modes(submatrix(acl, LONG_STATES)) if m.is_oscillatory),
               key=lambda m: m.wn)


def test_aft_cg_needs_more_alpha_feedback(cg_schedule):
    forward = cg_schedule.gains(500.0, 0.0, 0.30)
    aft = cg_schedule.gains(500.0, 0.0, 0.38)
    assert aft.k_alpha > forward.k_alpha


def test_gains_between_design_cgs_are_between_their_values(cg_schedule):
    lo = cg_schedule.gains(500.0, 0.0, 0.34).k_alpha
    mid = cg_schedule.gains(500.0, 0.0, 0.36).k_alpha
    hi = cg_schedule.gains(500.0, 0.0, 0.38).k_alpha
    assert lo < mid < hi


@pytest.mark.parametrize("xcg", [0.30, 0.32, 0.36, 0.38])
def test_short_period_held_across_cg_range(cg_schedule, xcg):
    sp = short_period(500.0, 0.0, xcg, cg_schedule.gains(500.0, 0.0, xcg))
    assert sp.wn == pytest.approx(WN, abs=0.4)
    assert sp.zeta == pytest.approx(ZETA, abs=0.1)