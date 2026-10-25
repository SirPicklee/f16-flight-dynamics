"""
Trim for steady maneuvers: coordinated level turn and symmetric pull-up.

Coordinated turn: constant airspeed, altitude and turn rate, with no lateral
acceleration (the ball centred). Body rates follow from the turn rate and the
attitude, so the Euler angle rates match a steady turn exactly.

Pull-up: wings level at the bottom of a vertical circle (flight path angle
zero at that instant), with a prescribed constant pitch rate.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from f16 import constants as c
from f16.atmosphere import adc
from f16.dynamics import (ALPHA, ALT, BETA, PHI, POW, P, Q, R, THETA, VT,
                          aero_coefficients, f16_derivatives)
from f16.engine import tgear, thrust


@dataclass
class ManeuverTrim:
    x: np.ndarray
    u: np.ndarray
    residual: float
    success: bool

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.x[ALPHA]))

    @property
    def beta_deg(self) -> float:
        return float(np.degrees(self.x[BETA]))

    @property
    def bank_deg(self) -> float:
        return float(np.degrees(self.x[PHI]))

    @property
    def load_factor(self) -> float:
        """Load factor n = (aerodynamic lift + thrust component normal to the
        flight path) / weight, the quantity a level turn and a pull-up fix
        kinematically."""
        mach, qbar = adc(self.x[VT], self.x[ALT])
        cx, _, cz, _, _, _ = aero_coefficients(
            self.x[VT], self.alpha_deg, self.beta_deg, self.x[P], self.x[Q], self.x[R],
            self.u[1], self.u[2], self.u[3], c.XCGR)
        alpha = self.x[ALPHA]
        lift = qbar * c.S * (cx * np.sin(alpha) - cz * np.cos(alpha))
        t = thrust(self.x[POW], self.x[ALT], mach)
        return float((lift + t * np.sin(alpha)) / c.WEIGHT)


def _turn_state(z, vt, alt, turn_rate):
    throttle, el, ail, rdr, alpha, beta, phi, theta = z
    x = np.zeros(13)
    x[VT], x[ALPHA], x[BETA] = vt, alpha, beta
    x[PHI], x[THETA], x[ALT] = phi, theta, alt
    # Body rates of a steady turn about the vertical axis
    x[P] = -turn_rate * np.sin(theta)
    x[Q] = turn_rate * np.sin(phi) * np.cos(theta)
    x[R] = turn_rate * np.cos(phi) * np.cos(theta)
    x[POW] = tgear(throttle)
    return x, np.array([throttle, el, ail, rdr])


def trim_coordinated_turn(vt: float, alt: float, turn_rate_deg_s: float,
                          xcg: float = c.XCGR) -> ManeuverTrim:
    """Steady, level, coordinated turn at the given turn rate [deg/s]."""
    w = np.radians(turn_rate_deg_s)

    def residuals(z):
        x, u = _turn_state(z, vt, alt, w)
        xd = f16_derivatives(x, u, xcg)
        cy = aero_coefficients(vt, np.degrees(x[ALPHA]), np.degrees(x[BETA]),
                               x[P], x[Q], x[R], u[1], u[2], u[3], xcg)[1]
        return np.array([xd[VT], 100 * xd[ALPHA], 100 * xd[BETA],
                         100 * xd[P], 100 * xd[Q], 100 * xd[R],
                         xd[ALT], 100 * cy])          # level and coordinated

    bank_guess = min(np.arctan(vt * w / c.G), np.radians(84.0))
    z0 = np.array([0.3, -2.0, 0.0, 0.0, np.radians(5.0), 0.0, bank_guess, np.radians(3.0)])
    lo = [0.0, -25.0, -21.5, -30.0, np.radians(-10), np.radians(-30), -np.radians(85), -np.pi / 2]
    hi = [1.0, 25.0, 21.5, 30.0, np.radians(45), np.radians(30), np.radians(85), np.pi / 2]
    sol = least_squares(residuals, z0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    x, u = _turn_state(sol.x, vt, alt, w)
    res = float(np.linalg.norm(residuals(sol.x)))
    return ManeuverTrim(x, u, res, bool(sol.success and res < 1e-6))


def trim_pull_up(vt: float, alt: float, pitch_rate_deg_s: float,
                 xcg: float = c.XCGR) -> ManeuverTrim:
    """Wings-level pull-up at the bottom of the arc with constant pitch rate."""
    q = np.radians(pitch_rate_deg_s)

    def build(z):
        throttle, el, alpha = z
        x = np.zeros(13)
        x[VT], x[ALPHA], x[THETA], x[ALT] = vt, alpha, alpha, alt  # gamma = 0
        x[Q], x[POW] = q, tgear(throttle)
        return x, np.array([throttle, el, 0.0, 0.0])

    def residuals(z):
        x, u = build(z)
        xd = f16_derivatives(x, u, xcg)
        return np.array([xd[VT], 100 * xd[ALPHA], 100 * xd[Q]])

    z0 = np.array([0.3, -3.0, np.radians(5.0)])
    bounds = ([0.0, -25.0, np.radians(-10)], [1.0, 25.0, np.radians(45)])
    sol = least_squares(residuals, z0, bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    x, u = build(sol.x)
    res = float(np.linalg.norm(residuals(sol.x)))
    return ManeuverTrim(x, u, res, bool(sol.success and res < 1e-6))


G_LIMIT = 9.0  # structural load factor limit


def max_sustained_turn_rate(vt: float, alt: float, xcg: float = c.XCGR,
                            alpha_limit_deg: float = 25.0, tol: float = 0.05) -> float:
    """Largest steady level turn rate [deg/s] the aircraft can hold at full
    throttle or below, within the angle-of-attack and load factor limits.
    Returns 0 if even level flight cannot be trimmed."""

    def ok(rate):
        t = trim_coordinated_turn(vt, alt, rate, xcg)
        return (t.success and t.alpha_deg <= alpha_limit_deg
                and t.load_factor <= G_LIMIT)

    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, 40.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo