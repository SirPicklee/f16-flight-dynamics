"""
Time-domain simulation of the nonlinear model and its linearization.

Both models are driven by the same control input history so their
responses can be compared directly.
"""

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from f16 import constants as c
from f16.dynamics import f16_derivatives

ControlFn = Callable[[float], np.ndarray]


def doublet(t: float, start: float, width: float, amplitude: float) -> float:
    """Square doublet: +amplitude for `width` seconds, then -amplitude."""
    if start <= t < start + width:
        return amplitude
    if start + width <= t < start + 2.0 * width:
        return -amplitude
    return 0.0


def simulate_nonlinear(x0: np.ndarray, control: ControlFn, t_final: float,
                       dt: float = 0.01, xcg: float = c.XCGR):
    """Integrate the nonlinear model. Returns (t, X) with X of shape (N, 13)."""
    t_eval = np.arange(0.0, t_final + dt / 2, dt)
    sol = solve_ivp(lambda t, x: f16_derivatives(x, control(t), xcg),
                    (0.0, t_final), x0, t_eval=t_eval,
                    max_step=dt, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


def simulate_linear(a: np.ndarray, b: np.ndarray, x0: np.ndarray, u0: np.ndarray,
                    control: ControlFn, t_final: float, dt: float = 0.01):
    """Integrate dx/dt = A dx + B du about (x0, u0). Returns (t, X) in absolute values."""
    t_eval = np.arange(0.0, t_final + dt / 2, dt)
    sol = solve_ivp(lambda t, dx: a @ dx + b @ (control(t) - u0),
                    (0.0, t_final), np.zeros(len(x0)), t_eval=t_eval,
                    max_step=dt, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T + x0