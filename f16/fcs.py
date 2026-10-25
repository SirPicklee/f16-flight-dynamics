"""
Flight control system with actuator dynamics and a yaw-damper washout.

Each control surface is driven by a first-order actuator with position and
rate limits. The yaw damper feeds back washed-out yaw rate, so it damps the
dutch roll but does not oppose a steady turn.

Augmented state z (17):
    z[0:13]  aircraft state (see dynamics.py)
    z[13]    elevator position [deg]
    z[14]    aileron position  [deg]
    z[15]    rudder position   [deg]
    z[16]    washout filter state [rad/s]
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from f16 import constants as c
from f16.dynamics import ALPHA, Q, R, f16_derivatives
from f16.linearize import LAT_STATES, LONG_STATES
from f16.sas import SasGains

N_AIRCRAFT = 13
DE, DA, DR, XW = 13, 14, 15, 16


@dataclass(frozen=True)
class Actuator:
    bandwidth: float       # [rad/s]
    position_limit: float  # [deg]
    rate_limit: float      # [deg/s]

    def rate(self, position: float, command: float) -> float:
        target = np.clip(command, -self.position_limit, self.position_limit)
        return float(np.clip(self.bandwidth * (target - position),
                             -self.rate_limit, self.rate_limit))


# F-16 surface actuators (Stevens & Lewis)
ELEVATOR = Actuator(20.2, 25.0, 60.0)
AILERON = Actuator(20.2, 21.5, 80.0)
RUDDER = Actuator(20.2, 30.0, 120.0)

PilotFn = Callable[[float], np.ndarray]


class FlightControlSystem:
    """SAS with actuators and washout, wrapped around the nonlinear model."""

    def __init__(self, x0: np.ndarray, u0: np.ndarray, gains: SasGains,
                 washout_tau: float = 1.0, xcg: float = c.XCGR):
        self.x0, self.u0, self.g = x0, u0, gains
        self.tau, self.xcg = washout_tau, xcg

    def initial_state(self) -> np.ndarray:
        z = np.zeros(17)
        z[:N_AIRCRAFT] = self.x0
        z[DE], z[DA], z[DR] = self.u0[1], self.u0[2], self.u0[3]
        return z

    def commands(self, z: np.ndarray, pilot: np.ndarray) -> np.ndarray:
        """Surface commands [deg] from trim, feedback and pilot input."""
        dx = z[:N_AIRCRAFT] - self.x0
        r_washed = z[R] - z[XW]
        de = (self.u0[1] + self.g.k_alpha * np.degrees(dx[ALPHA])
              + self.g.k_q * np.degrees(dx[Q]) + pilot[1])
        da = self.u0[2] + pilot[2]
        dr = self.u0[3] + self.g.k_r * np.degrees(r_washed) + pilot[3]
        return np.array([de, da, dr])

    def derivatives(self, t: float, z: np.ndarray, pilot: PilotFn) -> np.ndarray:
        p = pilot(t)
        cmd = self.commands(z, p)
        u = np.array([self.u0[0] + p[0], z[DE], z[DA], z[DR]])
        zd = np.zeros(17)
        zd[:N_AIRCRAFT] = f16_derivatives(z[:N_AIRCRAFT], u, self.xcg)
        zd[DE] = ELEVATOR.rate(z[DE], cmd[0])
        zd[DA] = AILERON.rate(z[DA], cmd[1])
        zd[DR] = RUDDER.rate(z[DR], cmd[2])
        zd[XW] = (z[R] - z[XW]) / self.tau
        return zd

    def simulate(self, pilot: PilotFn, t_final: float, dt: float = 0.01):
        """Returns (t, Z) with Z of shape (N, 17)."""
        t_eval = np.arange(0.0, t_final + dt / 2, dt)
        sol = solve_ivp(lambda t, z: self.derivatives(t, z, pilot),
                        (0.0, t_final), self.initial_state(), t_eval=t_eval,
                        max_step=dt, rtol=1e-8, atol=1e-10)
        return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# Linear analysis of the augmented loops
# ---------------------------------------------------------------------------

def pitch_loop(a: np.ndarray, b: np.ndarray, g: SasGains):
    """Linear pitch loop broken at the elevator command.

    States [vt, alpha, theta, q, elevator]. Returns (Aa, Ba, K) such that the
    loop closes as d(elevator_cmd) = K z.
    """
    lon = np.array(LONG_STATES)
    aa = np.zeros((5, 5))
    aa[:4, :4] = a[np.ix_(lon, lon)]
    aa[:4, 4] = b[lon, 1]
    aa[4, 4] = -ELEVATOR.bandwidth
    ba = np.zeros((5, 1))
    ba[4, 0] = ELEVATOR.bandwidth
    k = np.zeros((1, 5))
    k[0, 1] = np.degrees(g.k_alpha)
    k[0, 3] = np.degrees(g.k_q)
    return aa, ba, k


def lateral_loop(a: np.ndarray, b: np.ndarray, g: SasGains, washout_tau: float = 1.0):
    """Linear yaw-damper loop broken at the rudder command.

    States [beta, phi, p, r, aileron, rudder, washout]. Returns (Aa, Ba, K).
    """
    lat = np.array(LAT_STATES)
    aa = np.zeros((7, 7))
    aa[:4, :4] = a[np.ix_(lat, lat)]
    aa[:4, 4] = b[lat, 2]
    aa[:4, 5] = b[lat, 3]
    aa[4, 4] = -AILERON.bandwidth
    aa[5, 5] = -RUDDER.bandwidth
    aa[6, 3] = 1.0 / washout_tau
    aa[6, 6] = -1.0 / washout_tau
    ba = np.zeros((7, 1))
    ba[5, 0] = RUDDER.bandwidth
    k = np.zeros((1, 7))
    k[0, 3] = np.degrees(g.k_r)
    k[0, 6] = -np.degrees(g.k_r)
    return aa, ba, k


@dataclass
class Margins:
    phase_margin_deg: float
    crossover: float          # [rad/s]
    delay_margin: float       # [s]
    gain_reduction_db: float  # how far the loop gain may drop before instability


def loop_transfer(aa: np.ndarray, ba: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Loop transfer L(jw) = -K (jwI - Aa)^-1 Ba for a loop broken at the actuator input."""
    n = aa.shape[0]
    m = 1j * w[:, None, None] * np.eye(n) - aa          # one matrix per frequency
    x = np.linalg.solve(m, np.broadcast_to(ba, (len(w), n, 1)))
    return -(k @ x)[:, 0, 0]


def loop_margins(aa: np.ndarray, ba: np.ndarray, k: np.ndarray) -> Margins:
    """Phase, delay and gain-reduction margins of the loop broken at the actuator input."""
    w = np.logspace(-2, 3, 20000)
    loop = loop_transfer(aa, ba, k, w)
    i = int(np.where(np.diff(np.sign(np.abs(loop) - 1.0)) != 0)[0][-1])
    pm = 180.0 + float(np.degrees(np.angle(loop[i])))
    wc = float(w[i])

    def max_real(scale):
        return float(np.max(np.linalg.eigvals(aa + scale * ba @ k).real))

    # Smallest gain scale that still stabilizes the loop (bisection on [0, 1])
    if max_real(0.0) < 0.0:
        gain_red = -np.inf               # open loop already stable
    else:
        lo, hi = 0.0, 1.0                # unstable at lo, stable at hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if max_real(mid) < 0.0 else (mid, hi)
        gain_red = 20.0 * np.log10(hi)
    return Margins(pm, wc, np.radians(pm) / wc, float(gain_red))