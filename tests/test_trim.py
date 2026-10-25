import pytest

from f16.trim import trim_level


def test_recovers_reference_trim_point():
    r = trim_level(502.0, alt=0.0)
    assert r.success
    assert r.throttle == pytest.approx(0.1385, abs=5e-4)
    assert r.elevator_deg == pytest.approx(-0.7588, abs=0.01)
    assert r.alpha_deg == pytest.approx(2.115, abs=0.01)


def test_angle_of_attack_decreases_with_speed():
    alphas = [trim_level(v).alpha_deg for v in [300.0, 400.0, 500.0, 600.0, 700.0]]
    assert all(a > b for a, b in zip(alphas, alphas[1:]))


def test_back_side_of_power_curve():
    # Below the minimum-drag speed, flying slower needs MORE throttle
    t300 = trim_level(300.0).throttle
    t400 = trim_level(400.0).throttle
    t600 = trim_level(600.0).throttle
    assert t300 > t400
    assert t600 > t400


def test_forward_cg_needs_more_nose_up_elevator():
    els = [trim_level(502.0, xcg=x).elevator_deg for x in [0.30, 0.35, 0.38]]
    assert els[0] < els[1] < els[2]


def test_climb_needs_more_throttle():
    level = trim_level(502.0).throttle
    climb = trim_level(502.0, gamma_deg=5.0).throttle
    assert climb > level


def test_trim_at_altitude():
    r = trim_level(700.0, alt=30000.0)
    assert r.success
    assert r.residual < 1e-6