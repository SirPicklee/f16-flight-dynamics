import numpy as np
import pytest

from f16.envelope import survey, survey_point


@pytest.fixture(scope="module")
def coarse_grid():
    speeds = np.arange(250.0, 951.0, 50.0)
    altitudes = np.arange(0.0, 45001.0, 5000.0)
    return [p for row in survey(speeds, altitudes) for p in row if p.feasible]


def test_reference_point_matches_earlier_analysis():
    p = survey_point(502.0, 0.0)
    assert p.feasible
    assert p.alpha_deg == pytest.approx(2.1215, abs=1e-3)
    assert p.max_real_long == pytest.approx(0.0978, abs=1e-3)
    assert p.time_to_double == pytest.approx(7.08, abs=0.05)


def test_too_slow_at_high_altitude_is_outside_envelope():
    assert not survey_point(300.0, 40000.0).feasible


def test_mach_limit_bounds_the_envelope():
    assert survey_point(950.0, 0.0).feasible
    assert not survey_point(1050.0, 0.0).feasible


def test_low_alpha_trim_is_divergent(coarse_grid):
    band = [p.max_real_long for p in coarse_grid if 0.5 < p.alpha_deg < 4.5]
    assert band and min(band) > 0.05


def test_alpha_5_to_10_deg_is_neutrally_stable(coarse_grid):
    band = [p.max_real_long for p in coarse_grid if 5.5 < p.alpha_deg < 9.5]
    assert band and max(band) < 0.005


def test_dutch_roll_damping_falls_with_altitude():
    assert survey_point(600.0, 30000.0).dutch_roll_zeta < survey_point(600.0, 0.0).dutch_roll_zeta