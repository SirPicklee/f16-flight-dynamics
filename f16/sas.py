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
from scipy.optimize import fsolve

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


def design_pitch_sas(a: np.ndarray, b: np.ndarray,
                     wn: float = 3.0, zeta: float = 0.7) -> SasGains:
    """Choose k_alpha, k_q so the short period has the requested wn and zeta."""
    target = _target_root(wn, zeta)

    def error(k):
        acl = closed_loop_matrix(a, b, SasGains(k_alpha=k[0], k_q=k[1]))
        roots = np.linalg.eigvals(submatrix(acl, LONG_STATES))
        nearest = roots[np.argmin(np.abs(roots - target))]
        return [nearest.real - target.real, nearest.imag - target.imag]

    k_alpha, k_q = fsolve(error, x0=[2.0, 0.5])
    return SasGains(k_alpha=float(k_alpha), k_q=float(k_q))


def design_yaw_damper(a: np.ndarray, b: np.ndarray, zeta: float = 0.5) -> float:
    """Choose k_r so the dutch roll has the requested damping ratio."""

    def error(k):
        acl = closed_loop_matrix(a, b, SasGains(k_r=k[0]))
        dr = [m for m in modes(submatrix(acl, LAT_STATES)) if m.is_oscillatory]
        return [dr[0].zeta - zeta]

    (k_r,) = fsolve(error, x0=[0.5])
    return float(k_r)