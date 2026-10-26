"""List every failing Monte Carlo sample of the CG-scheduled SAS and the
requirement it misses, with the open-loop instability it had to overcome."""

import numpy as np

from f16.linearize import LONG_STATES, linearize, submatrix
from f16.monte_carlo import (GAIN_REDUCTION_REQUIRED, PM_REQUIRED,
                             ZETA_SP_RANGE, run)
from f16.schedule import build_cg_schedule
from f16.trim import trim_level

N_SAMPLES = 300
SEED = 1
WN = 2.5


def failed_requirements(r) -> str:
    reasons = []
    if not r.stable:
        reasons.append("unstable")
    if r.phase_margin < PM_REQUIRED:
        reasons.append("PM")
    if r.gain_reduction_db > GAIN_REDUCTION_REQUIRED:
        reasons.append("GM")
    if not ZETA_SP_RANGE[0] <= r.sp_zeta <= ZETA_SP_RANGE[1]:
        reasons.append("zeta")
    return "+".join(reasons)


def open_loop_root(s) -> tuple[float, float]:
    """Trim angle of attack [deg] and largest open-loop pitch root [1/s]."""
    t = trim_level(s.vt, alt=s.alt, xcg=s.xcg)
    a, _ = linearize(t.x, t.u, xcg=s.xcg)
    return t.alpha_deg, float(np.max(np.linalg.eigvals(submatrix(a, LONG_STATES)).real))


def main():
    print(f"Running {N_SAMPLES} samples with the CG-scheduled SAS (about a minute)...")
    cg_schedule = build_cg_schedule(wn=WN)
    results = run(N_SAMPLES, cg_schedule.gains, seed=SEED)
    fails = sorted((r for r in results if not r.passed), key=lambda r: r.sample.xcg)

    print(f"\n{len(fails)} failing samples, sorted by CG\n")
    print("  xcg    alt[ft]  VT[ft/s] alpha  OL root  delay  act.bw   Mq    PM    GM[dB]  zeta  fails")
    for r in fails:
        s = r.sample
        alpha, root = open_loop_root(s)
        print(f"  {s.xcg:.3f} {s.alt:8.0f} {s.vt:7.0f}  {alpha:5.1f}  {root:+6.3f}  "
              f"{1000 * s.delay:4.0f}ms {s.actuator_bw:5.1f}  {s.mq_scale:5.2f} "
              f"{r.phase_margin:5.1f}  {r.gain_reduction_db:6.1f}  {r.sp_zeta:5.2f}  "
              f"{failed_requirements(r)}")

    low_delay = [r for r in fails if r.sample.delay <= 0.02]
    print(f"\nFailures with delay <= 20 ms: {len(low_delay)}")
    for reason in ("PM", "GM", "zeta", "unstable"):
        n = sum(reason in failed_requirements(r).split("+") for r in low_delay)
        print(f"  {reason:8s}: {n}")


if __name__ == "__main__":
    main()