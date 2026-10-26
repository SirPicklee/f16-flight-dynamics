"""
Normal-acceleration (g) command system for the pitch axis.

The pilot commands an increment in load factor; a PI controller on the nz
error, plus an optional feedforward of the command, drives the elevator on
top of the pitch SAS:

    elevator_cmd = SAS(alpha, q) + kff * r - kp * (b * r - nz) - ki * integral(r - nz)

with r the commanded and nz the measured load factor increment. The PI terms
are negative because positive elevator pitches the nose down. Feedforward and
setpoint weight act outside the loop: they change the response but not the
closed-loop poles or margins.

Proportional nz feedback does not add damping on this airframe: the direct
tail lift of the elevator makes it shift a closed-loop pole towards the
right. The recommended design (design_g_command) therefore uses kp = 0 and
chooses the alpha and q feedback together with the integral gain.

A sustained g command curves the flight path without end, so on the full
longitudinal model the slow speed and path modes have no meaningful
equilibrium (they are left to the pilot). The loop is therefore designed on
the short-period model, where a sustained g increment is a steady pitch rate.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import StateSpace, step

from f16 import constants as c
from f16.dynamics import load_factor_nz
from f16.fcs import pitch_loop
from f16.linearize import LONG_STATES
from f16.sas import SasGains

SHORT_PERIOD_KEEP = np.array([1, 3, 4])   # alpha, q, elevator in [vt, alpha, theta, q, elevator]


def nz_output_row(x0: np.ndarray, u0: np.ndarray, xcg: float = c.XCGR) -> np.ndarray:
    """d(nz)/d[vt, alpha, theta, q, elevator] at the trim point (elevator in deg)."""
    row = np.zeros(5)
    steps = (0.1, 1e-4, 1e-4, 1e-4)
    for j, (i, h) in enumerate(zip(LONG_STATES, steps)):
        dx = np.zeros(13)
        dx[i] = h
        row[j] = (load_factor_nz(x0 + dx, u0, xcg)
                  - load_factor_nz(x0 - dx, u0, xcg)) / (2.0 * h)
    du = np.zeros(4)
    du[1] = 1e-3
    row[4] = (load_factor_nz(x0, u0 + du, xcg)
              - load_factor_nz(x0, u0 - du, xcg)) / 2e-3
    return row


def _augmented_plant(a, b, x0, u0, g, xcg, short_period):
    """SAS-augmented pitch dynamics (acl, ba) and nz output row cz."""
    aa, ba, k = pitch_loop(a, b, g)
    acl = aa + ba @ k
    cz = nz_output_row(x0, u0, xcg)
    if short_period:
        keep = SHORT_PERIOD_KEEP
        acl, ba, cz = acl[np.ix_(keep, keep)], ba[keep], cz[keep]
    return acl, ba, cz


def feedforward_gain(a: np.ndarray, b: np.ndarray, x0: np.ndarray, u0: np.ndarray,
                     g: SasGains, xcg: float = c.XCGR) -> float:
    """Elevator [deg] per g of steady nz increment for the SAS-augmented
    short-period model (negative: nose-up elevator)."""
    acl, ba, cz = _augmented_plant(a, b, x0, u0, g, xcg, short_period=True)
    static_gain = float(-(cz @ np.linalg.solve(acl, ba))[0])   # g per deg
    return 1.0 / static_gain


def nz_command_loop(a: np.ndarray, b: np.ndarray, x0: np.ndarray, u0: np.ndarray,
                    g: SasGains, kp: float, ki: float, xcg: float = c.XCGR,
                    short_period: bool = True, kff: float = 0.0,
                    setpoint_weight: float = 1.0):
    """Closed-loop linear model from commanded nz increment to nz increment.

    States [alpha, q, elevator, integral of nz error] with short_period=True,
    or [vt, alpha, theta, q, elevator, integral] for the full longitudinal model.
    Returns (M, Bm, Cm). kp in deg/g, ki in deg/(g s), kff in deg/g.
    """
    acl, ba, cz = _augmented_plant(a, b, x0, u0, g, xcg, short_period)
    cz = cz[None, :]
    n = acl.shape[0]
    m = np.zeros((n + 1, n + 1))
    m[:n, :n] = acl + kp * ba @ cz
    m[:n, n] = -ki * ba[:, 0]
    m[n, :n] = -cz[0]
    bm = np.zeros((n + 1, 1))
    bm[:n, 0] = (kff - setpoint_weight * kp) * ba[:, 0]
    bm[n, 0] = 1.0
    cm = np.zeros((1, n + 1))
    cm[0, :n] = cz[0]
    return m, bm, cm


def design_g_command(a: np.ndarray, b: np.ndarray, x0: np.ndarray, u0: np.ndarray,
                     wn: float = 3.0, zeta: float = 0.7, integral_pole: float = 3.0,
                     xcg: float = c.XCGR, guess=(0.7, 0.2, 5.0)) -> tuple[SasGains, float]:
    """Joint design of the alpha and q feedback and the nz integral gain (kp = 0).

    Places the dominant closed-loop pair at wn, zeta and the integral pole at
    -integral_pole on the short-period model. Returns (SasGains, ki).
    """
    target = complex(-zeta * wn, wn * np.sqrt(1.0 - zeta ** 2))

    def error(k):
        g = SasGains(k_alpha=k[0], k_q=k[1])
        m, _, _ = nz_command_loop(a, b, x0, u0, g, 0.0, k[2], xcg)
        roots = np.linalg.eigvals(m)
        pair = roots[np.argmin(np.abs(roots - target))]
        real = roots[np.argmin(np.abs(roots + integral_pole))]
        return [pair.real - target.real, pair.imag - target.imag, real.real + integral_pole]

    k_alpha, k_q, ki = fsolve(error, x0=list(guess))
    return SasGains(k_alpha=float(k_alpha), k_q=float(k_q)), float(ki)


@dataclass
class StepMetrics:
    stable: bool
    dc_gain: float
    rise_time: float       # 10 % to 90 % [s]
    overshoot: float       # [%]
    settling_time: float   # last time outside +/-5 % [s]
    initial_dip: float     # most negative early response [g]


def step_metrics(m: np.ndarray, bm: np.ndarray, cm: np.ndarray,
                 t_final: float = 4.0) -> StepMetrics:
    """Response of nz to a 1 g step command."""
    stable = bool(np.max(np.linalg.eigvals(m).real) < 0.0)
    dc = float(-(cm @ np.linalg.solve(m, bm))[0, 0])
    if not stable:
        return StepMetrics(False, dc, np.nan, np.nan, np.nan, np.nan)
    t = np.linspace(0.0, t_final, 4001)
    _, y = step(StateSpace(m, bm, cm, np.zeros((1, 1))), T=t)
    y = np.asarray(y).ravel()
    above10, above90 = np.where(y >= 0.1)[0], np.where(y >= 0.9)[0]
    rise = t[above90[0]] - t[above10[0]] if len(above90) else np.nan
    outside = np.where(np.abs(y - 1.0) > 0.05)[0]
    settle = t[outside[-1]] if len(outside) else 0.0
    return StepMetrics(True, dc, float(rise), float(max(0.0, 100.0 * (y.max() - 1.0))),
                       float(settle), float(min(0.0, y[t < 0.5].min())))