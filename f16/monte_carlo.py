"""
Monte Carlo robustness analysis of the pitch SAS.

Each sample draws a flight condition inside the envelope, a CG position and
a set of uncertainties on the airframe, actuator and computing delay. The
pitch loop is rebuilt for that sample and checked against requirements:

    * closed loop stable
    * phase margin of at least 45 deg
    * at least 6 dB of gain reduction before instability
    * short-period damping between 0.35 and 1.30 (MIL-F-8785C, Level 1)

CG changes are applied to the nonlinear model (re-trim and re-linearize);
the other uncertainties scale terms of the linearized pitch dynamics.
The aerodynamic model of the "true" aircraft can be chosen independently of
the model the gains were designed on. An optional lead filter can be placed
on the SAS output.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from f16.atmosphere import adc
from f16.dynamics import ALPHA, Q
from f16.envelope import ALPHA_LIMIT_DEG, MACH_LIMIT
from f16.fcs import loop_margins
from f16.linearize import LONG_STATES, linearize, modes
from f16.sas import EL, SasGains
from f16.trim import trim_level

PM_REQUIRED = 45.0               # deg
GAIN_REDUCTION_REQUIRED = -6.0   # dB
ZETA_SP_RANGE = (0.35, 1.30)


@dataclass
class Sample:
    vt: float           # [ft/s]
    alt: float          # [ft]
    xcg: float          # CG position [fraction of chord]
    mq_scale: float     # pitch damping derivative factor
    mde_scale: float    # elevator effectiveness factor
    jyy_scale: float    # pitch inertia factor
    actuator_bw: float  # [rad/s]
    delay: float        # computing and sensor delay [s]


@dataclass(frozen=True)
class LeadFilter:
    """Lead filter C(s) = (p/z) (s + z) / (s + p) with unit gain at low frequency."""
    center: float   # frequency of maximum phase lead [rad/s]
    ratio: float    # pole / zero ratio, >= 1

    @property
    def zero(self) -> float:
        return self.center / np.sqrt(self.ratio)

    @property
    def pole(self) -> float:
        return self.center * np.sqrt(self.ratio)

    @property
    def max_phase_deg(self) -> float:
        return float(np.degrees(np.arcsin((self.ratio - 1.0) / (self.ratio + 1.0))))


@dataclass
class Result:
    sample: Sample
    stable: bool
    sp_wn: float
    sp_zeta: float
    phase_margin: float
    gain_reduction_db: float

    @property
    def passed(self) -> bool:
        return (self.stable
                and self.phase_margin >= PM_REQUIRED
                and self.gain_reduction_db <= GAIN_REDUCTION_REQUIRED
                and ZETA_SP_RANGE[0] <= self.sp_zeta <= ZETA_SP_RANGE[1])


def draw_sample(rng: np.random.Generator, cg_range=(0.30, 0.38), model: str = "tables"):
    """Draw one sample with a flight condition inside the envelope.
    Returns the sample and its trim result."""
    while True:
        vt, alt = rng.uniform(250.0, 950.0), rng.uniform(0.0, 45000.0)
        xcg = rng.uniform(*cg_range)
        if adc(vt, alt)[0] > MACH_LIMIT:
            continue
        r = trim_level(vt, alt=alt, xcg=xcg, model=model)
        if r.success and r.alpha_deg <= ALPHA_LIMIT_DEG:
            break
    s = Sample(vt, alt, xcg,
               mq_scale=rng.normal(1.0, 0.15),
               mde_scale=rng.normal(1.0, 0.10),
               jyy_scale=rng.uniform(0.9, 1.1),
               actuator_bw=rng.uniform(15.0, 25.0),
               delay=rng.uniform(0.0, 0.05))
    return s, r


def perturbed_pitch_loop(a: np.ndarray, b: np.ndarray, g: SasGains, s: Sample,
                         lead: LeadFilter | None = None):
    """Pitch loop [vt, alpha, theta, q, elevator, delay (, lead)] broken at the SAS output.

    The delay is a first-order Pade approximation (1 - s*tau/2) / (1 + s*tau/2).
    With a lead filter, the filter state x_f obeys x_f' = -p x_f + K x and the
    SAS output is (p/z) (K x + (z - p) x_f).
    """
    lon = np.array(LONG_STATES)
    a4 = a[np.ix_(lon, lon)].copy()
    b4 = b[lon, EL].copy() * s.mde_scale
    iq = LONG_STATES.index(Q)
    a4[iq, iq] *= s.mq_scale
    a4[iq, :] /= s.jyy_scale
    b4[iq] /= s.jyy_scale

    w, tau = s.actuator_bw, max(s.delay, 1e-6)
    aa = np.zeros((6, 6))
    aa[:4, :4] = a4
    aa[:4, 4] = b4
    # Actuator driven by the delayed command y = 2*x_d - u
    aa[4, 4] = -w
    aa[4, 5] = 2.0 * w
    aa[5, 5] = -2.0 / tau
    ba = np.zeros((6, 1))
    ba[4, 0] = -w
    ba[5, 0] = 2.0 / tau
    k = np.zeros((1, 6))
    k[0, LONG_STATES.index(ALPHA)] = np.degrees(g.k_alpha)
    k[0, iq] = np.degrees(g.k_q)
    if lead is None:
        return aa, ba, k

    z, p = lead.zero, lead.pole
    n = aa.shape[0]
    aa2 = np.zeros((n + 1, n + 1))
    aa2[:n, :n] = aa
    aa2[n, :n] = k[0]          # filter driven by the proportional SAS signal
    aa2[n, n] = -p
    ba2 = np.vstack([ba, [[0.0]]])
    k2 = np.zeros((1, n + 1))
    k2[0, :n] = (p / z) * k[0]
    k2[0, n] = (p / z) * (z - p)
    return aa2, ba2, k2


def evaluate(s: Sample, r, gains: SasGains, model: str = "tables",
             lead: LeadFilter | None = None) -> Result:
    """Rebuild the pitch loop for one sample and measure it."""
    a, b = linearize(r.x, r.u, xcg=s.xcg, model=model)
    aa, ba, k = perturbed_pitch_loop(a, b, gains, s, lead)
    acl = aa + ba @ k
    stable = bool(np.max(np.linalg.eigvals(acl).real) < 0.0)
    osc = [m for m in modes(acl) if m.is_oscillatory and m.wn < 15.0]
    sp = max(osc, key=lambda m: m.wn) if osc else None
    m = loop_margins(aa, ba, k)
    return Result(s, stable, sp.wn if sp else np.nan, sp.zeta if sp else np.nan,
                  m.phase_margin_deg, m.gain_reduction_db)


GainSource = Callable[[float, float, float], SasGains]


def run(n: int, gain_source: GainSource, seed: int = 0, cg_range=(0.30, 0.38),
        model: str = "tables", lead: LeadFilter | None = None) -> list[Result]:
    """Run n samples; gain_source(vt, alt, xcg) returns the SAS gains to use.
    `model` is the aerodynamic model of the aircraft being tested."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        s, r = draw_sample(rng, cg_range, model)
        results.append(evaluate(s, r, gain_source(s.vt, s.alt, s.xcg), model, lead))
    return results


def pass_rate(results: list[Result], max_xcg: float = np.inf,
              max_delay: float = np.inf) -> tuple[int, int]:
    """Count passing samples among those within the given CG and delay limits.
    Returns (passed, total)."""
    subset = [r for r in results
              if r.sample.xcg <= max_xcg and r.sample.delay <= max_delay]
    return sum(r.passed for r in subset), len(subset)


def failure_rate_upper_bound(n_samples: int) -> float:
    """95 % upper confidence bound on the failure rate when no failures were
    seen in n_samples (the 'rule of three')."""
    return 3.0 / n_samples if n_samples > 0 else 1.0