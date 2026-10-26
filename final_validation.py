"""Validate the final pitch SAS design on unseen samples and on both aero models."""

from diagnose_failures import failed_requirements
from f16.monte_carlo import LeadFilter, run
from f16.schedule import build_cg_schedule

N_SAMPLES = 300
SEED = 3                                   # never used during design
FINAL_LEAD = LeadFilter(center=6.0, ratio=2.5)


def main():
    print("Building the optimal CG schedule (a few minutes)...")
    schedule = build_cg_schedule(wn="optimal")
    print(f"Final design: optimal CG schedule + lead at {FINAL_LEAD.center:.0f} rad/s, "
          f"ratio {FINAL_LEAD.ratio:.1f}\n")

    for model in ("tables", "morelli"):
        for lead, name in ((None, "without lead"), (FINAL_LEAD, "with lead")):
            res = run(N_SAMPLES, schedule.gains, seed=SEED, model=model, lead=lead)
            fails = [r for r in res if not r.passed]
            print(f"{model:8s} {name:13s}: {len(res) - len(fails)} of {len(res)} pass")
            if lead is not None:
                for r in sorted(fails, key=lambda r: r.sample.xcg):
                    s = r.sample
                    print(f"    xcg {s.xcg:.3f}  alt {s.alt:6.0f}  VT {s.vt:4.0f}  "
                          f"delay {1000 * s.delay:2.0f} ms  PM {r.phase_margin:5.1f}  "
                          f"GM {r.gain_reduction_db:6.1f}  {failed_requirements(r)}")
        print()


if __name__ == "__main__":
    main()