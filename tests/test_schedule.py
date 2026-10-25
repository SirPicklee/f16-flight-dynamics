import numpy as np
import pytest

from f16.linearize import LONG_STATES, linearize, modes, submatrix
from f16.sas import SasGains, closed_loop_matrix
from f16.schedule import build_schedule
from f16.trim import trim_level


@pytest.fixture(scope="module")
def schedule():
    return build_schedule()


def short_period(a, b, g):
    acl = closed_loop_matrix(a, b, g)
    return max((m for m in modes(submatrix(acl, LONG_STATES)) if m.is_oscillatory),
               key=lambda m: m.wn)


def test_schedule_reproduces_nominal_design_on_grid(schedule):
    g = schedule.gains(500.0, 0.0)
    r = trim_level(500.0)
    a, b = linearize(r.x, r.u)
    sp = short_period(a, b, g)
    assert sp.wn == pytest.approx(3.0, abs=1e-3)
    assert sp.zeta == pytest.approx(0.7, abs=1e-3)


def test_gains_fall_with_dynamic_pressure(schedule):
    slow, fast = schedule.gains(300.0, 0.0), schedule.gains(800.0, 0.0)
    assert slow.k_alpha > fast.k_alpha
    assert slow.k_q > fast.k_q
    assert slow.k_r > fast.k_r


@pytest.mark.parametrize("vt, alt", [(375.0, 7500.0), (625.0, 22500.0), (725.0, 37500.0)])
def test_scheduled_gains_hold_short_period_between_grid_points(schedule, vt, alt):
    r = trim_level(vt, alt=alt)
    a, b = linearize(r.x, r.u)
    sp = short_period(a, b, schedule.gains(vt, alt))
    assert sp.wn == pytest.approx(3.0, abs=0.35)
    assert sp.zeta == pytest.approx(0.7, abs=0.08)


def test_scheduling_beats_fixed_gains_at_high_altitude(schedule):
    fixed = SasGains(k_alpha=0.7121, k_q=0.2007, k_r=0.6619)
    r = trim_level(725.0, alt=37500.0)
    a, b = linearize(r.x, r.u)
    sp_fixed = short_period(a, b, fixed)
    sp_sched = short_period(a, b, schedule.gains(725.0, 37500.0))
    assert abs(sp_sched.wn - 3.0) < abs(sp_fixed.wn - 3.0)