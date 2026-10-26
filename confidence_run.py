"""Monte Carlo restricted to CG 0.30-0.33: does the derived CG limit hold,
and how tight is the confidence bound?"""

from diagnose_failures import failed_requirements
from f16.monte_carlo import failure_rate_upper_bound, run
from f16.schedule import build_cg_schedule

N_SAMPLES = 300
SEED = 2          # a different seed: new samples, not the ones the limit came from
CG_RANGE = (0.30, 0.33)
WN = 2.5


def main():
    print(f"Running {N_SAMPLES} samples with CG in {CG_RANGE} (about a minute)...")
    cg_schedule = build_cg_schedule(wn=WN)
    results = run(N_SAMPLES, cg_schedule.gains, seed=SEED, cg_range=CG_RANGE)
    fails = [r for r in results if not r.passed]

    print(f"\n{len(results) - len(fails)} of {len(results)} samples pass")
    if not fails:
        bound = 100.0 * failure_rate_upper_bound(len(results))
        print(f"No failures: failure rate < {bound:.1f} % with 95 % confidence")
    else:
        print("\nFailing samples:")
        print("  xcg    alt[ft]  VT[ft/s]  delay   PM    GM[dB]  fails")
        for r in fails:
            s = r.sample
            print(f"  {s.xcg:.3f} {s.alt:8.0f} {s.vt:7.0f}  {1000 * s.delay:4.0f}ms "
                  f"{r.phase_margin:5.1f}  {r.gain_reduction_db:6.1f}  {failed_requirements(r)}")


if __name__ == "__main__":
    main()