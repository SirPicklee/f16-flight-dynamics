import pytest

from f16.aero_tables import CX_TABLE, cm, cx, cz


def test_exact_grid_point():
    # alpha = 0 deg, el = 0 deg -> table row 2, column 2
    assert cx(0.0, 0.0) == pytest.approx(CX_TABLE[2, 2])


def test_midpoint_is_average():
    expected = 0.5 * (cx(0.0, 0.0) + cx(5.0, 0.0))
    assert cx(2.5, 0.0) == pytest.approx(expected)


def test_linear_extrapolation_above_table():
    # last two points: -2.248 (40 deg), -2.229 (45 deg)
    # at 50 deg the trend continues: -2.229 + 0.019 = -2.210
    assert cz(50.0, 0.0, 0.0) == pytest.approx(-2.210)


def test_pitch_moment_near_zero_at_reference_trim():
    # Reference level-flight trim (sea level, 502 ft/s, xcg = 0.35):
    # alpha ~ 2.1 deg, elevator ~ -0.76 deg -> Cm should be ~ 0
    assert cm(2.1, -0.76) == pytest.approx(0.0, abs=1e-3)