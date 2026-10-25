"""
F-16 aerodynamic coefficient tables (longitudinal).

Data: NASA TP-1538 (Nguyen et al., 1979) as tabulated in Stevens & Lewis,
Aircraft Control and Simulation, 2nd ed., Appendix A. Table values follow
the Python port in AeroBenchVVPython (GPL-3.0).

Angles are in degrees. Outside the table range the lookup extrapolates
linearly from the nearest segment, matching the reference implementation.
"""

import numpy as np

ALPHA_GRID = np.arange(-10.0, 50.0, 5.0)  # -10 ... 45 deg (12 points)
EL_GRID = np.arange(-24.0, 36.0, 12.0)    # -24 ... 24 deg (5 points)

# Rows: alpha, columns: elevator
CX_TABLE = np.array([
    [-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166],
    [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167],
    [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138],
    [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091],
    [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040],
]).T

CM_TABLE = np.array([
    [.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192],
    [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093],
    [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, .000, -.013, .032],
    [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006],
    [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005],
]).T

CZ_TABLE = np.array(
    [.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229]
)


def _segment(grid: np.ndarray, x: float) -> tuple[int, float]:
    """Find the grid segment for x and the fractional position within it.

    The segment index is clamped to the first/last segment, so a fraction
    outside [0, 1] gives linear extrapolation beyond the table edges.
    """
    i = int(np.searchsorted(grid, x, side="right")) - 1
    i = min(max(i, 0), len(grid) - 2)
    frac = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, frac


def interp1(grid: np.ndarray, table: np.ndarray, x: float) -> float:
    """1-D linear interpolation with linear extrapolation."""
    i, f = _segment(grid, x)
    return float(table[i] + f * (table[i + 1] - table[i]))


def interp2(grid1: np.ndarray, grid2: np.ndarray, table: np.ndarray,
            x1: float, x2: float) -> float:
    """2-D bilinear interpolation with linear extrapolation."""
    i, f = _segment(grid1, x1)
    j, h = _segment(grid2, x2)
    lo = table[i, j] + f * (table[i + 1, j] - table[i, j])
    hi = table[i, j + 1] + f * (table[i + 1, j + 1] - table[i, j + 1])
    return float(lo + h * (hi - lo))


def cx(alpha: float, el: float) -> float:
    """Body x-axis force coefficient. alpha, el in deg."""
    return interp2(ALPHA_GRID, EL_GRID, CX_TABLE, alpha, el)


def cz(alpha: float, beta: float, el: float) -> float:
    """Body z-axis force coefficient. alpha, beta, el in deg."""
    base = interp1(ALPHA_GRID, CZ_TABLE, alpha)
    return base * (1.0 - (beta / 57.3) ** 2) - 0.19 * (el / 25.0)


def cm(alpha: float, el: float) -> float:
    """Pitching moment coefficient. alpha, el in deg."""
    return interp2(ALPHA_GRID, EL_GRID, CM_TABLE, alpha, el)