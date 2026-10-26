import numpy as np
import pytest

from f16 import constants as c
from f16.dynamics import (ALPHA, BETA, PHI, P, Q, R, THETA, VT,
                          load_factor_nz)
from f16.maneuver_trim import trim_coordinated_turn, trim_pull_up
from f16.trim import trim_level


def kinematic_nz(x):
    """nz from the body z force equation with w_dot = 0."""
    uu = x[VT] * np.cos(x[ALPHA]) * np.cos(x[BETA])
    vv = x[VT] * np.sin(x[BETA])
    return (np.cos(x[THETA]) * np.cos(x[PHI])
            + (x[Q] * uu - x[P] * vv) / c.G)


def test_level_flight_nz_equals_cos_theta():
    r = trim_level(502.0)
    assert load_factor_nz(r.x, r.u) == pytest.approx(np.cos(r.x[THETA]), abs=1e-6)


@pytest.mark.parametrize("rate", [5.0, 10.0, 15.0])
def test_pull_up_nz_matches_kinematics(rate):
    t = trim_pull_up(502.0, 0.0, rate)
    assert load_factor_nz(t.x, t.u) == pytest.approx(kinematic_nz(t.x), rel=1e-4)


@pytest.mark.parametrize("rate", [5.0, 10.0])
def test_turn_nz_matches_kinematics(rate):
    t = trim_coordinated_turn(502.0, 0.0, rate)
    assert load_factor_nz(t.x, t.u) == pytest.approx(kinematic_nz(t.x), rel=1e-4)


def test_pulling_up_increases_nz():
    level = trim_level(502.0)
    pull = trim_pull_up(502.0, 0.0, 10.0)
    assert load_factor_nz(pull.x, pull.u) > load_factor_nz(level.x, level.u) + 2.0