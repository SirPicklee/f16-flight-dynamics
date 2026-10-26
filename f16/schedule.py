"""
Gain scheduling of the SAS over the flight envelope.

SAS gains are designed at every feasible point of a speed-altitude grid and
stored as tables. In flight, the gains are interpolated from the tables using
the current airspeed and altitude, as a real flight control computer does
with its air data.

CgGainSchedule adds the CG position as a third scheduling variable: one
speed-altitude table is built per CG value and the gains are interpolated
linearly between them, as a flight control computer can do with a CG
estimate from the fuel system.
"""

from dataclasses import dataclass

import numpy as np

from f16 import constants as c
from f16.aero_tables import interp2
from f16.envelope import survey_point
from f16.linearize import linearize
from f16.sas import SasGains, design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

DEFAULT_SPEEDS = np.arange(250.0, 951.0, 50.0)       # ft/s
DEFAULT_ALTITUDES = np.arange(0.0, 45001.0, 5000.0)  # ft
DEFAULT_CGS = (0.30, 0.34, 0.38)


def _fill_outside(table: np.ndarray) -> np.ndarray:
    """Copy the nearest feasible value along each altitude row into NaN cells,
    so interpolation near the envelope edge stays well defined."""
    out = table.copy()
    for row in out:
        valid = np.where(~np.isnan(row))[0]
        if len(valid) == 0:
            continue
        for j in np.where(np.isnan(row))[0]:
            row[j] = row[valid[np.argmin(np.abs(valid - j))]]
    return out


@dataclass
class GainSchedule:
    speeds: np.ndarray
    altitudes: np.ndarray
    k_alpha: np.ndarray   # tables indexed [altitude, speed]
    k_q: np.ndarray
    k_r: np.ndarray
    feasible: np.ndarray  # True where the gains were designed

    def gains(self, vt: float, alt: float, xcg: float | None = None) -> SasGains:
        """Interpolated gains at the given airspeed [ft/s] and altitude [ft].
        The CG argument is accepted for a common interface and ignored."""
        args = (self.altitudes, self.speeds)
        return SasGains(k_alpha=interp2(*args, self.k_alpha, alt, vt),
                        k_q=interp2(*args, self.k_q, alt, vt),
                        k_r=interp2(*args, self.k_r, alt, vt))


def build_schedule(speeds=DEFAULT_SPEEDS, altitudes=DEFAULT_ALTITUDES,
                   wn: float = 3.0, zeta_sp: float = 0.7, zeta_dr: float = 0.5,
                   xcg: float = c.XCGR) -> GainSchedule:
    """Design pitch SAS and yaw damper gains at every feasible grid point."""
    shape = (len(altitudes), len(speeds))
    ka, kq, kr = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    feasible = np.zeros(shape, dtype=bool)
    for i, alt in enumerate(altitudes):
        for j, vt in enumerate(speeds):
            if not survey_point(vt, alt, xcg).feasible:
                continue
            r = trim_level(vt, alt=alt, xcg=xcg)
            a, b = linearize(r.x, r.u, xcg=xcg)
            g = design_pitch_sas(a, b, wn=wn, zeta=zeta_sp)
            ka[i, j], kq[i, j] = g.k_alpha, g.k_q
            kr[i, j] = design_yaw_damper(a, b, zeta=zeta_dr)
            feasible[i, j] = True
    return GainSchedule(np.asarray(speeds, float), np.asarray(altitudes, float),
                        _fill_outside(ka), _fill_outside(kq), _fill_outside(kr), feasible)


@dataclass
class CgGainSchedule:
    cgs: np.ndarray                   # CG values the tables were designed for
    schedules: list[GainSchedule]     # one speed-altitude schedule per CG

    def gains(self, vt: float, alt: float, xcg: float | None = None) -> SasGains:
        """Gains interpolated in speed, altitude and CG. Outside the CG range
        the nearest table is used."""
        if xcg is None:
            xcg = c.XCGR
        per_cg = [s.gains(vt, alt) for s in self.schedules]
        return SasGains(
            k_alpha=float(np.interp(xcg, self.cgs, [g.k_alpha for g in per_cg])),
            k_q=float(np.interp(xcg, self.cgs, [g.k_q for g in per_cg])),
            k_r=float(np.interp(xcg, self.cgs, [g.k_r for g in per_cg])))


def build_cg_schedule(cgs=DEFAULT_CGS, **design) -> CgGainSchedule:
    """Build one speed-altitude schedule per CG value. Keyword arguments
    (wn, zeta_sp, ...) are passed to build_schedule."""
    schedules = [build_schedule(xcg=x, **design) for x in cgs]
    return CgGainSchedule(np.asarray(cgs, float), schedules)