"""Does a lead filter on the pitch SAS lift the phase-margin ceiling at aft CG?"""

from f16.monte_carlo import (GAIN_REDUCTION_REQUIRED, PM_REQUIRED,
                             ZETA_SP_RANGE, LeadFilter, run)
from f16.schedule import build_cg_schedule

N_SAMPLES = 300
SEED = 1
CONFIGS = [None,
           LeadFilter(center=3.0, ratio=1.5),
           LeadFilter(center=3.0, ratio=2.5),
           LeadFilter(center=6.0, ratio=1.5),
           LeadFilter(center=6.0, ratio=2.5)]


def describe(lead):
    if lead is None:
        return "no lead filter"
    return (f"lead at {lead.center:.0f} rad/s, ratio {lead.ratio:.1f} "
            f"(+{lead.max_phase_deg:.0f} deg)")


def main():
    print("Building the optimal CG schedule (a few minutes)...")
    schedule = build_cg_schedule(wn="optimal")
    print(f"Running {N_SAMPLES} samples for {len(CONFIGS)} configurations "
          f"(about {len(CONFIGS)} minutes)...\n")

    print(f"{'configuration':42s} pass   PM   GM  zeta   aft-CG PM failures fixed")
    stubborn = None
    for lead in CONFIGS:
        res = run(N_SAMPLES, schedule.gains, seed=SEED, lead=lead)
        if stubborn is None:   # baseline: remember the aft-CG phase-margin failures
            stubborn = [i for i, r in enumerate(res)
                        if not r.passed and r.sample.xcg >= 0.36
                        and r.phase_margin < PM_REQUIRED]
        fails = [r for r in res if not r.passed]
        n_pm = sum(r.phase_margin < PM_REQUIRED for r in fails)
        n_gm = sum(r.gain_reduction_db > GAIN_REDUCTION_REQUIRED for r in fails)
        n_z = sum(not ZETA_SP_RANGE[0] <= r.sp_zeta <= ZETA_SP_RANGE[1] for r in fails)
        fixed = sum(res[i].passed for i in stubborn)
        print(f"{describe(lead):42s} {len(res) - len(fails):4d} {n_pm:4d} {n_gm:4d} {n_z:5d}"
              f"   {fixed:3d} / {len(stubborn)}")


if __name__ == "__main__":
    main()