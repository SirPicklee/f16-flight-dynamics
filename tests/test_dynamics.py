import numpy as np
import pytest

from f16 import constants as c
from f16.dynamics import (ALPHA, BETA, NORTH, PHI, POW, PSI, P, Q, R, VT,
                          f16_derivatives, inertia_coefficients)
from f16.engine import tgear

# Reference level-flight trim (sea level, 502 ft/s, xcg = 0.35)
TRIM_THROTTLE = 0.1385
TRIM_ELEVATOR = -0.7588  # deg
TRIM_ALPHA = 0.03691     # rad


def trim_state() -> np.ndarray:
    x = np.zeros(13)
    x[VT] = 502.0
    x[ALPHA] = TRIM_ALPHA
    x[4] = TRIM_ALPHA            # theta = alpha in level flight
    x[POW] = tgear(TRIM_THROTTLE)
    return x


def trim_control() -> np.ndarray:
    return np.array([TRIM_THROTTLE, TRIM_ELEVATOR, 0.0, 0.0])


def test_inertia_coefficients_match_reference_values():
    book = [-0.770, 0.02755, 1.055e-4, 1.642e-6, 0.9604,
            1.759e-2, 1.792e-5, -0.7336, 1.587e-5]
    derived = inertia_coefficients(c.JXX, c.JYY, c.JZZ, c.JXZ)
    for d, b in zip(derived, book):
        assert d == pytest.approx(b, rel=5e-4)


def test_reference_trim_point_is_in_equilibrium():
    xd = f16_derivatives(trim_state(), trim_control())
    assert abs(xd[VT]) < 1e-2      # ft/s^2
    assert abs(xd[ALPHA]) < 1e-3   # rad/s
    assert abs(xd[Q]) < 1e-4       # rad/s^2
    assert abs(xd[POW]) < 1e-6     # engine settled
    assert xd[NORTH] == pytest.approx(502.0)
    assert abs(xd[11]) < 1e-9      # no climb or descent


def test_symmetric_flight_has_no_lateral_motion():
    xd = f16_derivatives(trim_state(), trim_control())
    for i in [BETA, PHI, PSI, P, R]:
        assert xd[i] == 0.0


def test_more_power_accelerates_the_aircraft():
    x = trim_state()
    x[POW] = 30.0
    assert f16_derivatives(x, trim_control())[VT] > 0.0


def test_trailing_edge_down_elevator_pitches_nose_down():
    u = trim_control()
    u[1] = TRIM_ELEVATOR + 5.0
    assert f16_derivatives(trim_state(), u)[Q] < 0.0