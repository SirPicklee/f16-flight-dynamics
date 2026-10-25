import pytest

from f16.engine import pdot, tgear, thrust


def test_throttle_gearing_end_points():
    assert tgear(0.0) == pytest.approx(0.0)
    assert tgear(1.0) == pytest.approx(100.0)


def test_throttle_gearing_nearly_continuous_at_military():
    # Both branches should give about 50 % at the military power detent
    assert tgear(0.77) == pytest.approx(50.0, abs=0.02)
    assert tgear(0.7701) == pytest.approx(50.0, abs=0.05)


def test_power_lag_is_zero_at_steady_state():
    assert pdot(30.0, 30.0) == 0.0
    assert pdot(80.0, 80.0) == 0.0


def test_sea_level_static_thrust():
    assert thrust(0.0, 0.0, 0.0) == pytest.approx(1060.0)    # idle
    assert thrust(50.0, 0.0, 0.0) == pytest.approx(12680.0)  # military
    assert thrust(100.0, 0.0, 0.0) == pytest.approx(20000.0) # max afterburner


def test_military_thrust_drops_with_altitude():
    values = [thrust(50.0, h, 0.4) for h in [0.0, 10000.0, 20000.0, 30000.0]]
    assert all(a > b for a, b in zip(values, values[1:]))