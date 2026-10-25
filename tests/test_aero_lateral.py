import numpy as np
import pytest

from f16.aero_tables import ALPHA_GRID, DAMPING_NAMES, cl, cn, damping


def test_no_rolling_or_yawing_moment_without_sideslip():
    for alpha in [0.0, 10.0, 30.0]:
        assert cl(alpha, 0.0) == 0.0
        assert cn(alpha, 0.0) == 0.0


def test_lateral_coefficients_are_antisymmetric_in_beta():
    for alpha in [-5.0, 7.5, 22.0]:
        for beta in [3.0, 12.0, 25.0]:
            assert cl(alpha, -beta) == pytest.approx(-cl(alpha, beta))
            assert cn(alpha, -beta) == pytest.approx(-cn(alpha, beta))


def test_dihedral_effect_at_cruise():
    # Positive sideslip should roll the aircraft away from the sideslip
    assert cl(5.0, 5.0) < 0.0


def test_weathercock_stability_at_low_alpha():
    # Positive sideslip should yaw the nose back into the wind
    assert cn(5.0, 5.0) > 0.0


def test_directional_instability_at_high_alpha():
    # Known F-16 characteristic: directional stability is lost at high alpha
    assert cn(40.0, 5.0) < 0.0


def test_pitch_and_roll_damping_always_negative():
    for alpha in ALPHA_GRID:
        d = dict(zip(DAMPING_NAMES, damping(float(alpha))))
        assert d["Cmq"] < 0.0
        assert d["Clp"] < 0.0