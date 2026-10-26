"""
Roll rate command system.

The pilot commands a roll rate; the aileron command combines feedforward of
the command with proportional feedback of the roll rate error:

    aileron = kff * p_cmd + kp * (p_cmd - p)       [deg, p in deg/s]

On the one-state roll mode model  p_dot = Lp p + Lda aileron  the closed-loop
pole is Lp - Lda * kp, and kff = -Lp / Lda gives zero steady-state error.
Positive aileron gives a negative rolling moment on this model, so both gains
come out negative.
"""

from dataclasses import dataclass

import numpy as np

from f16.dynamics import P

AILERON = 2   # index in the control vector


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