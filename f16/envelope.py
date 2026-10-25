"""
Flight envelope survey: trim and open-loop stability over speed and altitude.

A grid point is inside the envelope when a level-flight trim exists with
throttle at or below full power, angle of attack within the limiter value
and Mach number within the range covered by the data.
"""

from dataclasses import dataclass

import numpy as np

from f16 import constants as c
from f16.atmosphere import adc
from f16.linearize import LAT_STATES, LONG_STATES, linearize, modes, submatrix
from f16.trim import trim_level

ALPHA_LIMIT_DEG = 25.0   # angle-of-attack limiter of the F-16 flight control system
MACH_LIMIT = 0.9         # stay subsonic, within the thrust and aero data


@dataclass
class EnvelopePoint:
    vt: float
    alt: float
    mach: float
    qbar: float
    feasible: bool
    alpha_deg: float = np.nan
    throttle: float = np.nan
    elevator_deg: float = np.nan
    max_real_long: float = np.nan   # largest real part of longitudinal roots [1/s]
    dutch_roll_wn: float = np.nan
    dutch_roll_zeta: float = np.nan

    @property
    def time_to_double(self) -> float:
        """Time for the divergent pitch mode to double [s]; inf if stable."""
        if not self.feasible or self.max_real_long <= 0.0:
            return np.inf
        return float(np.log(2.0) / self.max_real_long)


def survey_point(vt: float, alt: float, xcg: float = c.XCGR) -> EnvelopePoint:
    """Trim and linearize at one condition, if it is inside the envelope."""
    mach, qbar = adc(vt, alt)
    r = trim_level(vt, alt=alt, xcg=xcg)
    feasible = r.success and r.alpha_deg <= ALPHA_LIMIT_DEG and mach <= MACH_LIMIT
    pt = EnvelopePoint(vt, alt, mach, qbar, feasible)
    if not feasible:
        return pt
    a, _ = linearize(r.x, r.u, xcg=xcg)
    long_roots = np.linalg.eigvals(submatrix(a, LONG_STATES))
    # Dutch roll is the highest-frequency lateral oscillation (at low speed the
    # roll and spiral roots can also merge into a slow lateral oscillation)
    dr = max((m for m in modes(submatrix(a, LAT_STATES)) if m.is_oscillatory),
             key=lambda m: m.wn)
    pt.alpha_deg, pt.throttle, pt.elevator_deg = r.alpha_deg, r.throttle, r.elevator_deg
    pt.max_real_long = float(np.max(long_roots.real))
    pt.dutch_roll_wn, pt.dutch_roll_zeta = dr.wn, dr.zeta
    return pt


def survey(speeds, altitudes, xcg: float = c.XCGR) -> list[list[EnvelopePoint]]:
    """Survey a grid. Result is indexed [altitude][speed]."""
    return [[survey_point(float(v), float(h), xcg) for v in speeds] for h in altitudes]


def field(grid: list[list[EnvelopePoint]], name: str) -> np.ndarray:
    """Extract one attribute as a 2-D array, NaN outside the envelope."""
    return np.array([[getattr(p, name) if p.feasible else np.nan for p in row]
                     for row in grid])