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
# ---------------------------------------------------------------------------
# Lateral-directional coefficients
# ---------------------------------------------------------------------------

BETA_ABS_GRID = np.arange(0.0, 35.0, 5.0)   # 0 ... 30 deg (7 points), |beta|
BETA_GRID = np.arange(-30.0, 40.0, 10.0)    # -30 ... 30 deg (7 points)

# Rows: alpha, columns: |beta|
CL_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-.001, -.004, -.008, -.012, -.016, -.022, -.022, -.021, -.015, -.008, -.013, -.015],
    [-.003, -.009, -.017, -.024, -.030, -.041, -.045, -.040, -.016, -.002, -.010, -.019],
    [-.001, -.010, -.020, -.030, -.039, -.054, -.057, -.054, -.023, -.006, -.014, -.027],
    [.000, -.010, -.022, -.034, -.047, -.060, -.069, -.067, -.033, -.036, -.035, -.035],
    [.007, -.010, -.023, -.034, -.049, -.063, -.081, -.079, -.060, -.058, -.062, -.059],
    [.009, -.011, -.023, -.037, -.050, -.068, -.089, -.088, -.091, -.076, -.077, -.076],
], dtype=float).T

CN_TABLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033],
    [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057],
    [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073],
    [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041],
    [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013],
    [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001],
], dtype=float).T

# Rows: alpha, columns: beta (-30 ... 30 deg)
DLDA_TABLE = np.array([
    [-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017],
    [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016],
    [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014],
    [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012],
    [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011],
    [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010],
    [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008],
]).T

DLDR_TABLE = np.array([
    [.005, .017, .014, .010, -.005, .009, .019, .005, -.000, -.005, -.011, .008],
    [.007, .016, .014, .014, .013, .009, .012, .005, .000, .004, .009, .007],
    [.013, .013, .011, .012, .011, .009, .008, .005, -.002, .005, .003, .005],
    [.018, .015, .015, .014, .014, .014, .014, .015, .013, .011, .006, .001],
    [.015, .014, .013, .013, .012, .011, .011, .010, .008, .008, .007, .003],
    [.021, .011, .010, .011, .010, .009, .008, .010, .006, .005, .000, .001],
    [.023, .010, .011, .011, .011, .010, .008, .010, .006, .014, .020, .000],
]).T

DNDA_TABLE = np.array([
    [.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016],
    [.002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011],
    [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006],
    [-.011, -.011, -.010, -.009, -.008, -.006, .000, .004, .007, .010, .004, .010],
    [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011],
    [-.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006],
    [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001],
]).T

DNDR_TABLE = np.array([
    [-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013],
    [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008],
    [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013],
    [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016],
    [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009],
    [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014],
    [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010],
]).T


def cy(beta: float, ail: float, rdr: float) -> float:
    """Side force coefficient. beta, ail, rdr in deg."""
    return -0.02 * beta + 0.021 * (ail / 20.0) + 0.086 * (rdr / 30.0)


def cl(alpha: float, beta: float) -> float:
    """Rolling moment coefficient due to sideslip (antisymmetric in beta)."""
    return float(np.sign(beta)) * interp2(ALPHA_GRID, BETA_ABS_GRID, CL_TABLE, alpha, abs(beta))


def cn(alpha: float, beta: float) -> float:
    """Yawing moment coefficient due to sideslip (antisymmetric in beta)."""
    return float(np.sign(beta)) * interp2(ALPHA_GRID, BETA_ABS_GRID, CN_TABLE, alpha, abs(beta))


def dlda(alpha: float, beta: float) -> float:
    """Rolling moment per unit aileron (aileron normalized by 20 deg)."""
    return interp2(ALPHA_GRID, BETA_GRID, DLDA_TABLE, alpha, beta)


def dldr(alpha: float, beta: float) -> float:
    """Rolling moment per unit rudder (rudder normalized by 30 deg)."""
    return interp2(ALPHA_GRID, BETA_GRID, DLDR_TABLE, alpha, beta)


def dnda(alpha: float, beta: float) -> float:
    """Yawing moment per unit aileron (aileron normalized by 20 deg)."""
    return interp2(ALPHA_GRID, BETA_GRID, DNDA_TABLE, alpha, beta)


def dndr(alpha: float, beta: float) -> float:
    """Yawing moment per unit rudder (rudder normalized by 30 deg)."""
    return interp2(ALPHA_GRID, BETA_GRID, DNDR_TABLE, alpha, beta)


# ---------------------------------------------------------------------------
# Damping derivatives
# ---------------------------------------------------------------------------

DAMPING_NAMES = ("CXq", "CYr", "CYp", "CZq", "Clr", "Clp", "Cmq", "Cnr", "Cnp")

# Rows: alpha, columns: derivatives in DAMPING_NAMES order
DAMPING_TABLE = np.array([
    [-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21],
    [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04],
    [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27],
    [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3],
    [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330],
    [-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100],
    [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00],
    [-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840],
    [.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150],
]).T


def damping(alpha: float) -> np.ndarray:
    """Damping derivatives at alpha [deg], ordered as DAMPING_NAMES."""
    return np.array([interp1(ALPHA_GRID, DAMPING_TABLE[:, i], alpha)
                     for i in range(len(DAMPING_NAMES))])