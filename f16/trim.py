"""
Trim: find steady wings-level flight conditions.

For a given airspeed, altitude and flight path angle, solve for the
throttle, elevator and angle of attack that make the longitudinal
accelerations zero (VT_dot = alpha_dot = q_dot = 0) with the engine settled.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from f16 import constants as c
from f16.dynamics import ALPHA, ALT, POW, Q, THETA, VT, f16_derivatives
from f16.engine import tgear


@dataclass
class TrimResult:
    x: np.ndarray        # trimmed state vector
    u: np.ndarray        # trimmed control vector
    residual: float      # norm of the remaining accelerations
    success: bool

    @property
    def throttle(self) -> float:
        return float(self.u[0])

    @property
    def elevator_deg(self) -> float:
        return float(self.u[1])

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.x[ALPHA]))


def _build(z: np.ndarray, vt: float, alt: float, gamma: float):
    """Map unknowns z = [throttle, elevator_deg, alpha_rad] to (x, u)."""
    throttle, elevator, alpha = z
    x = np.zeros(13)
    x[VT] = vt
    x[ALPHA] = alpha
    x[THETA] = alpha + gamma     # wings level, no sideslip
    x[ALT] = alt
    x[POW] = tgear(throttle)     # engine at steady state
    u = np.array([throttle, elevator, 0.0, 0.0])
    return x, u


def trim_level(vt: float, alt: float = 0.0, gamma_deg: float = 0.0,
               xcg: float = c.XCGR) -> TrimResult:
    """Trim for steady wings-level flight (climb if gamma_deg > 0)."""
    gamma = np.radians(gamma_deg)

    def residuals(z):
        x, u = _build(z, vt, alt, gamma)
        xd = f16_derivatives(x, u, xcg)
        # Scale so each residual is of similar size near the solution
        return np.array([xd[VT], 100.0 * xd[ALPHA], 100.0 * xd[Q]])

    z0 = np.array([0.2, -2.0, np.radians(5.0)])
    bounds = ([0.0, -25.0, np.radians(-10.0)], [1.0, 25.0, np.radians(45.0)])
    sol = least_squares(residuals, z0, bounds=bounds,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)

    x, u = _build(sol.x, vt, alt, gamma)
    res = float(np.linalg.norm(residuals(sol.x)))
    return TrimResult(x=x, u=u, residual=res, success=bool(sol.success and res < 1e-6))