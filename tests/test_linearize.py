import numpy as np
import pytest

from f16 import constants as c
from f16.dynamics import C4, C7, C9, PHI, P, Q, R, THETA
from f16.linearize import (LAT_STATES, LONG_STATES, linearize, modes,
                           submatrix)
from f16.trim import trim_level


@pytest.fixture(scope="module")
def nominal():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    return a, b


@pytest.fixture(scope="module")
def forward_cg():
    r = trim_level(502.0, xcg=0.30)
    a, _ = linearize(r.x, r.u, xcg=0.30)
    return a


def test_kinematic_rows_are_exact(nominal):
    a, _ = nominal
    assert a[THETA, Q] == pytest.approx(1.0)
    assert a[PHI, P] == pytest.approx(1.0)


def test_only_engine_gyroscopic_coupling_in_level_flight(nominal):
    # In symmetric flight the only link between pitch and roll/yaw is the
    # spinning engine rotor (HX). Everything else must be zero.
    a, _ = nominal
    lon, lat = np.array(LONG_STATES), np.array(LAT_STATES)
    a_lon_lat = a[np.ix_(lon, lat)].copy()
    a_lat_lon = a[np.ix_(lat, lon)].copy()

    assert a[Q, R] == pytest.approx(-C7 * c.HX)   # pitch accel from yaw rate
    assert a[P, Q] == pytest.approx(C4 * c.HX)    # roll accel from pitch rate
    assert a[R, Q] == pytest.approx(C9 * c.HX)    # yaw accel from pitch rate

    a_lon_lat[3, 3] = 0.0   # (q, r) entry
    a_lat_lon[2, 3] = 0.0   # (p, q) entry
    a_lat_lon[3, 3] = 0.0   # (r, q) entry
    assert np.max(np.abs(a_lon_lat)) < 1e-9
    assert np.max(np.abs(a_lat_lon)) < 1e-9


def test_nominal_cg_is_statically_unstable(nominal):
    # Relaxed static stability: one unstable real root in pitch
    a, _ = nominal
    roots = [m.eigenvalue for m in modes(submatrix(a, LONG_STATES))]
    unstable = [lam for lam in roots if lam.real > 0]
    assert len(unstable) == 1
    assert abs(unstable[0].imag) < 1e-9


def test_forward_cg_has_classic_short_period_and_phugoid(forward_cg):
    ms = modes(submatrix(forward_cg, LONG_STATES))
    assert len(ms) == 2 and all(m.is_oscillatory for m in ms)
    phugoid, short_period = ms  # sorted by natural frequency
    assert all(m.eigenvalue.real < 0 for m in ms)
    assert short_period.wn > 10 * phugoid.wn
    assert short_period.wn == pytest.approx(1.91, abs=0.05)
    assert short_period.zeta == pytest.approx(0.63, abs=0.03)
    assert phugoid.period == pytest.approx(85.0, abs=3.0)


def test_lateral_modes(nominal):
    a, _ = nominal
    ms = modes(submatrix(a, LAT_STATES))
    real = sorted([m for m in ms if not m.is_oscillatory], key=lambda m: m.wn)
    osc = [m for m in ms if m.is_oscillatory]
    spiral, roll = real
    (dutch_roll,) = osc
    assert 0.0 < roll.time_constant < 1.0
    assert spiral.time_constant > 20.0
    assert dutch_roll.wn == pytest.approx(3.09, abs=0.05)
    assert dutch_roll.zeta == pytest.approx(0.137, abs=0.01)


def test_elevator_effectiveness_sign(nominal):
    _, b = nominal
    assert b[Q, 1] < 0.0  # trailing edge down -> nose down pitch acceleration