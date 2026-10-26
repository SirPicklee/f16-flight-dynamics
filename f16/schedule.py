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

The short-period target can be fixed, or chosen at every grid point
("optimal") to maximize the worse of the phase and gain margins, normalized
by their requirements, for worst-case hardware.
"""

from dataclasses import dataclass

import numpy as np

from f16 import constants as c
from f16.aero_tables import interp2
from f16.envelope import survey_point
from f16.fcs import loop_margins
from f16.linearize import linearize
from f16.monte_carlo import (GAIN_REDUCTION_REQUIRED, PM_REQUIRED, Sample,
                             perturbed_pitch_loop)
from f16.sas import SasGains, design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

DEFAULT_SPEEDS = np.arange(250.0, 951.0, 50.0)       # ft/s
DEFAULT_ALTITUDES = np.arange(0.0, 45001.0, 5000.0)  # ft
DEFAULT_CGS = (0.30, 0.34, 0.38)
WN_CANDIDATES = np.arange(2.5, 4.51, 0.5)            # rad/s, lower limit for handling qualities

# Hardware assumed when the short-period target is chosen for margin
WORST_CASE_HARDWARE = Sample(vt=0.0, alt=0.0, xcg=0.0, mq_scale=1.0, mde_scale=1.0,
                             jyy_scale=1.0, actuator_bw=15.0, delay=0.05)


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


def margin_score(a: np.ndarray, b: np.ndarray, g: SasGains,
                 hardware: Sample = WORST_CASE_HARDWARE) -> float:
    """Worse of phase margin and gain-reduction margin, each divided by its
    requirement (1.0 = exactly meets both). -inf if the loop is unstable."""
    aa, ba, k = perturbed_pitch_loop(a, b, g, hardware)
    if np.max(np.linalg.eigvals(aa + ba @ k).real) >= 0.0:
        return -np.inf
    m = loop_margins(aa, ba, k)
    return min(m.phase_margin_deg / PM_REQUIRED,
               m.gain_reduction_db / GAIN_REDUCTION_REQUIRED)


def best_wn(a: np.ndarray, b: np.ndarray, zeta: float = 0.7,
            candidates=WN_CANDIDATES) -> float:
    """Short-period target that maximizes the margin score at this point."""
    scores = [margin_score(a, b, design_pitch_sas(a, b, wn=w, zeta=zeta)) for w in candidates]
    return float(candidates[int(np.argmax(scores))])


@dataclass
class GainSchedule:
    speeds: np.ndarray
    altitudes: np.ndarray
    k_alpha: np.ndarray   # tables indexed [altitude, speed]
    k_q: np.ndarray
    k_r: np.ndarray
    feasible: np.ndarray  # True where the gains were designed
    wn: np.ndarray | None = None   # short-period target used at each point

    def gains(self, vt: float, alt: float, xcg: float | None = None) -> SasGains:
        """Interpolated gains at the given airspeed [ft/s] and altitude [ft].
        The CG argument is accepted for a common interface and ignored."""
        args = (self.altitudes, self.speeds)
        return SasGains(k_alpha=interp2(*args, self.k_alpha, alt, vt),
                        k_q=interp2(*args, self.k_q, alt, vt),
                        k_r=interp2(*args, self.k_r, alt, vt))


def build_schedule(speeds=DEFAULT_SPEEDS, altitudes=DEFAULT_ALTITUDES,
                   wn: float | str = 3.0, zeta_sp: float = 0.7, zeta_dr: float = 0.5,
                   xcg: float = c.XCGR) -> GainSchedule:
    """Design pitch SAS and yaw damper gains at every feasible grid point.
    `wn` is a fixed short-period target [rad/s], or "optimal" to choose it
    at every point with best_wn."""
    shape = (len(altitudes), len(speeds))
    ka, kq, kr = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    wn_table = np.full(shape, np.nan)
    feasible = np.zeros(shape, dtype=bool)
    for i, alt in enumerate(altitudes):
        for j, vt in enumerate(speeds):
            if not survey_point(vt, alt, xcg).feasible:
                continue
            r = trim_level(vt, alt=alt, xcg=xcg)
            a, b = linearize(r.x, r.u, xcg=xcg)
            w = best_wn(a, b, zeta=zeta_sp) if wn == "optimal" else float(wn)
            g = design_pitch_sas(a, b, wn=w, zeta=zeta_sp)
            ka[i, j], kq[i, j], wn_table[i, j] = g.k_alpha, g.k_q, w
            kr[i, j] = design_yaw_damper(a, b, zeta=zeta_dr)
            feasible[i, j] = True
    return GainSchedule(np.asarray(speeds, float), np.asarray(altitudes, float),
                        _fill_outside(ka), _fill_outside(kq), _fill_outside(kr),
                        feasible, wn_table)


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


def build_cg_schedule(cgs=DEFAULT_CGS, wn=3.0, **design) -> CgGainSchedule:
    """Build one speed-altitude schedule per CG value.

    `wn` is the short-period target: one value for all CGs, one value per CG,
    or "optimal" to choose it at every grid point. Other keyword arguments
    (zeta_sp, zeta_dr, ...) are passed to build_schedule.
    """
    if isinstance(wn, str):
        wns = [wn] * len(cgs)
    else:
        wns = [float(w) for w in np.broadcast_to(np.asarray(wn, float), (len(cgs),))]
    schedules = [build_schedule(xcg=x, wn=w, **design) for x, w in zip(cgs, wns)]
    return CgGainSchedule(np.asarray(cgs, float), schedules)