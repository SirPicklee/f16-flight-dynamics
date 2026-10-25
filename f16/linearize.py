"""
Linearization about a trim point and modal analysis.

The nonlinear model dx/dt = f(x, u) is approximated near a trim point
(x0, u0) by d(dx)/dt = A dx + B du, with A = df/dx and B = df/du computed
by central finite differences.
"""

from dataclasses import dataclass

import numpy as np

from f16 import constants as c
from f16.dynamics import (ALPHA, BETA, PHI, P, Q, R, THETA, VT,
                          f16_derivatives)

LONG_STATES = (VT, ALPHA, THETA, Q)
LONG_NAMES = ("vt", "alpha", "theta", "q")
LAT_STATES = (BETA, PHI, P, R)
LAT_NAMES = ("beta", "phi", "p", "r")

# Perturbation sizes per state (ft/s for VT, rad for angles, ft, %)
_DX = np.array([0.1, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
                1e-4, 1e-4, 1e-4, 1.0, 1.0, 1.0, 0.01])
_DU = np.array([1e-4, 1e-3, 1e-3, 1e-3])


def linearize(x0: np.ndarray, u0: np.ndarray, xcg: float = c.XCGR):
    """Return (A, B) Jacobians of f16_derivatives at (x0, u0)."""
    n, m = len(x0), len(u0)
    a = np.zeros((n, n))
    b = np.zeros((n, m))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = _DX[j]
        a[:, j] = (f16_derivatives(x0 + dx, u0, xcg)
                   - f16_derivatives(x0 - dx, u0, xcg)) / (2.0 * _DX[j])
    for j in range(m):
        du = np.zeros(m)
        du[j] = _DU[j]
        b[:, j] = (f16_derivatives(x0, u0 + du, xcg)
                   - f16_derivatives(x0, u0 - du, xcg)) / (2.0 * _DU[j])
    return a, b


def submatrix(a: np.ndarray, states) -> np.ndarray:
    """Extract the block of A for the given state indices."""
    idx = np.array(states)
    return a[np.ix_(idx, idx)]


@dataclass
class Mode:
    eigenvalue: complex
    wn: float       # natural frequency [rad/s]
    zeta: float     # damping ratio [-]

    @property
    def is_oscillatory(self) -> bool:
        return abs(self.eigenvalue.imag) > 1e-9

    @property
    def period(self) -> float:
        """Damped period [s] (oscillatory modes)."""
        return 2.0 * np.pi / abs(self.eigenvalue.imag)

    @property
    def time_constant(self) -> float:
        """Time constant [s] (real modes); negative means divergent."""
        return -1.0 / self.eigenvalue.real


def modes(a: np.ndarray) -> list[Mode]:
    """Eigenvalues of A as modes, one entry per real root or complex pair."""
    out = []
    for lam in np.linalg.eigvals(a):
        if lam.imag < -1e-9:
            continue  # keep one root of each complex pair
        wn = abs(lam)
        zeta = -lam.real / wn if wn > 0 else 0.0
        out.append(Mode(complex(lam), float(wn), float(zeta)))
    return sorted(out, key=lambda md: md.wn)