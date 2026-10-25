import numpy as np
import pytest

from f16 import constants as c
from f16.maneuver_trim import (G_LIMIT, max_sustained_turn_rate,
                               trim_coordinated_turn, trim_pull_up)
from f16.trim import trim_level


def test_zero_rate_turn_is_level_flight():
    turn = trim_coordinated_turn(502.0, 0.0, 0.0)
    level = trim_level(502.0)
    assert turn.success
    assert turn.u[0] == pytest.approx(level.throttle, abs=1e-6)
    assert turn.u[1] == pytest.approx(level.elevator_deg, abs=1e-5)


@pytest.mark.parametrize("rate", [5.0, 10.0, 15.0])
def test_level_turn_load_factor_matches_kinematics(rate):
    # Level turn: lift must balance gravity and the centripetal acceleration
    t = trim_coordinated_turn(502.0, 0.0, rate)
    expected = np.hypot(1.0, 502.0 * np.radians(rate) / c.G)
    assert t.success
    assert t.load_factor == pytest.approx(expected, rel=1e-3)


def test_turn_is_coordinated():
    t = trim_coordinated_turn(502.0, 0.0, 10.0)
    assert abs(t.beta_deg) < 0.1


def test_tighter_turn_needs_more_throttle_and_alpha():
    slow = trim_coordinated_turn(502.0, 0.0, 5.0)
    fast = trim_coordinated_turn(502.0, 0.0, 15.0)
    assert fast.u[0] > slow.u[0]
    assert fast.alpha_deg > slow.alpha_deg


@pytest.mark.parametrize("rate", [5.0, 10.0, 15.0])
def test_pull_up_load_factor_matches_kinematics(rate):
    t = trim_pull_up(502.0, 0.0, rate)
    expected = 1.0 + 502.0 * np.radians(rate) / c.G
    assert t.success
    assert t.load_factor == pytest.approx(expected, rel=1e-4)


def test_sustained_turn_rate_falls_with_altitude():
    assert max_sustained_turn_rate(500.0, 20000.0) < max_sustained_turn_rate(500.0, 0.0)


def test_high_speed_turn_is_load_factor_limited():
    rate = max_sustained_turn_rate(900.0, 0.0)
    t = trim_coordinated_turn(900.0, 0.0, rate)
    assert t.load_factor == pytest.approx(G_LIMIT, abs=0.1)