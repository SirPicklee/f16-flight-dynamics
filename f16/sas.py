"""
Stability augmentation system (SAS).

Pitch SAS: angle-of-attack and pitch-rate feedback to the elevator, which
stabilizes the relaxed-static-stability airframe and sets the short-period
frequency and damping.

Yaw damper: yaw-rate feedback to the rudder, which raises dutch roll damping.

Gains are in degrees of surface deflection per degree (or deg/s) of the
fed-back variable. Surface positions are limited to their physical range.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

from f16.dynamics import ALPHA, Q, R
from f16.linearize import LAT_STATES, LONG_STATES, modes, submatrix

ELEVATOR_LIMIT = 25.0   # deg
AILERON_LIMIT = 21.5    # deg
RUDDER_LIMIT = 30.0     # deg

EL, AIL, RDR = 1, 2, 3  # control vector indices


@dataclass
class SasGains:
    k_alpha: float = 0.0   # elevator deg per deg alpha
    k_q: float = 0.0       # elevator deg per deg/s pitch rate
    k_r: float = 0.0       # rudder deg per deg/s yaw rate


def feedback_matrix(g: SasGains) -> np.ndarray:
    """Gain matrix K (4 x 13) such that du = K dx, in deg per rad."""
    k = np.zeros((4, 13))
    k[EL, ALPHA] = np.degrees(g.k_alpha)
    k[EL, Q] = np.degrees(g.k_q)
    k[RDR, R] = np.degrees(g.k_r)
    return k


def closed_loop_matrix(a: np.ndarray, b: np.ndarray, g: SasGains) -> np.ndarray:
    """A + B K for the linearized aircraft with the SAS engaged."""
    return a + b @ feedback_matrix(g)


def sas_control(x: np.ndarray, x0: np.ndarray, u0: np.ndarray,
                g: SasGains, pilot: np.ndarray | None = None) -> np.ndarray:
    """Surface commands from trim controls, SAS feedback and pilot input."""
    u = u0 + feedback_matrix(g) @ (x - x0)
    if pilot is not None:
        u = u + pilot
    u[EL] = np.clip(u[EL], -ELEVATOR_LIMIT, ELEVATOR_LIMIT)
    u[AIL] = np.clip(u[AIL], -AILERON_LIMIT, AILERON_LIMIT)
    u[RDR] = np.clip(u[RDR], -RUDDER_LIMIT, RUDDER_LIMIT)
    return u


def _target_root(wn: float, zeta: float) -> complex:
    return complex(-zeta * wn, wn * np.sqrt(1.0 - zeta ** 2))


def short_period_gains(a: np.ndarray, b: np.ndarray, wn: float, zeta: float):
    """Exact pole placement on the two-state short-period approximation.

    With states [alpha, q] and elevator feedback de = k_a*alpha + k_q*q, the
    closed-loop trace and determinant are linear in the gains, so the two
    conditions trace = -2*zeta*wn and det = wn^2 give a 2x2 linear system.
    Returns gains in deg per deg (deg/s).
    """
    idx = np.array([ALPHA, Q])
    asp = a[np.ix_(idx, idx)]
    bsp = b[idx, EL] * np.degrees(1.0)          # per rad of feedback -> per deg
    adj = np.array([[asp[1, 1], -asp[0, 1]], [-asp[1, 0], asp[0, 0]]])
    # trace(A + b k^T) = trace(A) + b . k ;  det(A + b k^T) = det(A) + k . (adj(A) b)
    m = np.vstack([bsp, adj @ bsp])
    rhs = np.array([-2.0 * zeta * wn - np.trace(asp), wn ** 2 - np.linalg.det(asp)])
    k_alpha, k_q = np.linalg.solve(m, rhs)
    return float(k_alpha), float(k_q)


def design_pitch_sas(a: np.ndarray, b: np.ndarray,
                     wn: float = 3.0, zeta: float = 0.7) -> SasGains:
    """Choose k_alpha, k_q so the short period of the full longitudinal model
    has the requested wn and zeta.

    The two-state short-period solution is used as the starting point, and
    the gains are then refined on the four-state model.
    """
    target = _target_root(wn, zeta)

    def error(k):
        acl = closed_loop_matrix(a, b, SasGains(k_alpha=k[0], k_q=k[1]))
        roots = np.linalg.eigvals(submatrix(acl, LONG_STATES))
        nearest = roots[np.argmin(np.abs(roots - target))]
        return [nearest.real - target.real, nearest.imag - target.imag]

    k_alpha, k_q = fsolve(error, x0=short_period_gains(a, b, wn, zeta))
    return SasGains(k_alpha=float(k_alpha), k_q=float(k_q))


def design_yaw_damper(a: np.ndarray, b: np.ndarray, zeta: float = 0.5,
                      k_max: float = 10.0) -> float:
    """Choose k_r so the dutch roll has the requested damping ratio.

    At low speed and high angle of attack yaw-rate feedback alone may not
    reach the target; the gain giving the highest damping is returned then.
    """

    def dutch_roll_zeta(k):
        acl = closed_loop_matrix(a, b, SasGains(k_r=k))
        osc = [m for m in modes(submatrix(acl, LAT_STATES)) if m.is_oscillatory]
        return max(osc, key=lambda m: m.wn).zeta if osc else 1.0

    ks = np.linspace(0.0, k_max, 201)
    zetas = np.array([dutch_roll_zeta(k) for k in ks])
    above = np.where(zetas >= zeta)[0]
    if len(above) == 0:
        return float(ks[np.argmax(zetas)])
    i = above[0]
    return float(brentq(lambda k: dutch_roll_zeta(k) - zeta, ks[i - 1], ks[i]))