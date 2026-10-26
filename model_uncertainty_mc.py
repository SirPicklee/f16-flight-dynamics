"""Monte Carlo of the CG-scheduled SAS (designed on the lookup tables) flown
on both aerodynamic models: are the high-speed failures real or a table artifact?"""

import numpy as np

from diagnose_failures import failed_requirements
from f16.monte_carlo import run
from f16.schedule import build_cg_schedule

N_SAMPLES = 300
SEED = 1
WN = "optimal"           # short-period target chosen at every grid point


def report(name, results):
    fails = [r for r in results if not r.passed]
    family_a = [r for r in fails if r.sample.vt > 750.0 and r.sample.alt < 15000.0]
    print(f"\n{name}: {len(results) - len(fails)} of {len(results)} pass, "
          f"{len(family_a)} failures at high speed and low altitude")
    print("  xcg    alt[ft]  VT[ft/s]  delay   PM    GM[dB]  fails")
    for r in sorted(fails, key=lambda r: r.sample.xcg):
        s = r.sample
        print(f"  {s.xcg:.3f} {s.alt:8.0f} {s.vt:7.0f}  {1000 * s.delay:4.0f}ms "
              f"{r.phase_margin:5.1f}  {r.gain_reduction_db:6.1f}  {failed_requirements(r)}")


def main():
    print("Building the CG schedule (a few minutes with the optimal target)...")
    cg_schedule = build_cg_schedule(wn=WN)          # designed on the tables
    for cg, sched in zip(cg_schedule.cgs, cg_schedule.schedules):
        w = sched.wn[sched.feasible]
        print(f"  CG {cg:.2f}: short-period target {w.min():.1f} - {w.max():.1f} rad/s, "
              f"median {np.median(w):.1f}")

    print(f"\nRunning {N_SAMPLES} samples on each aero model (about two minutes)...")
    for model in ("tables", "morelli"):
        report(f"Aircraft model: {model}", run(N_SAMPLES, cg_schedule.gains,
                                               seed=SEED, model=model))


if __name__ == "__main__":
    main()