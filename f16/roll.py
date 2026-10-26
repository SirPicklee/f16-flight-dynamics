"""
Roll rate command system and aileron-rudder interconnect.

The pilot commands a roll rate; the aileron command combines feedforward of
the command with proportional feedback of the roll rate error:

    aileron = kff * p_cmd + kp * (p_cmd - p)       [deg, p in deg/s]

On the one-state roll mode model  p_dot = Lp p + Lda aileron  the closed-loop
pole is Lp - Lda * kp, and kff = -Lp / Lda gives zero steady-state error.
Positive aileron gives a negative rolling moment on this model, so both gains
come out negative.

The aileron-rudder interconnect (ARI) adds rudder = k_ari * p_cmd, chosen so
that a steady roll about the velocity vector (r = p tan(alpha), beta = 0)
has zero yaw acceleration, cancelling adverse yaw and supplying the yaw rate.
"""

from dataclasses import dataclass

import numpy as np

from f16.dynamics import P, R

AILERON, RUDDER = 2, 3   # indices in the control vector


@dataclass
class RollModel:
    lp: float    # roll damping [1/s]
    lda: float   # aileron effectiveness [(deg/s^2) per deg]


def roll_mode(a: np.ndarray, b: np.ndarray) -> RollModel:
    """One-state roll mode model from the linearized aircraft."""
    return RollModel(lp=float(a[P, P]), lda=float(np.degrees(b[P, AILERON])))


def design_roll_command(a: np.ndarray, b: np.ndarray,
                        bandwidth: float = 6.0) -> tuple[float, float]:
    """Gains (kp, kff) [deg of aileron per deg/s] placing the closed-loop roll
    pole at -bandwidth with unit steady-state gain."""
    m = roll_mode(a, b)
    kp = (m.lp + bandwidth) / m.lda
    kff = -m.lp / m.lda
    return float(kp), float(kff)


def design_ari(a: np.ndarray, b: np.ndarray, alpha0: float, kff: float) -> float:
    """ARI gain [deg of rudder per deg/s of roll rate command].

    alpha0 is the trim angle of attack [rad], kff the aileron feedforward gain.
    Solves  N_p p + N_r p tan(alpha0) + N_da kff p_cmd + N_dr k_ari p_cmd = 0
    with p = p_cmd, p and r in rad/s, surfaces in deg.
    """
    per_deg_s = np.radians(1.0)                 # 1 deg/s of roll rate in rad/s
    yaw_from_rates = (a[R, P] + a[R, R] * np.tan(alpha0)) * per_deg_s
    yaw_from_aileron = b[R, AILERON] * kff
    return float(-(yaw_from_rates + yaw_from_aileron) / b[R, RUDDER])