"""Derive CG and delay limits under which the CG-scheduled SAS meets all
requirements, from a Monte Carlo run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.monte_carlo import failure_rate_upper_bound, pass_rate, run
from f16.schedule import build_cg_schedule

OUT = Path("figures")
N_SAMPLES = 300
SEED = 1
WN = 2.5
CG_LIMITS = np.round(np.arange(0.32, 0.381, 0.01), 2)
DELAY_LIMITS = np.array([0.010, 0.020, 0.030, 0.040, 0.050])   # s


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Running {N_SAMPLES} samples with the CG-scheduled SAS (about a minute)...")
    cg_schedule = build_cg_schedule(wn=WN)
    results = run(N_SAMPLES, cg_schedule.gains, seed=SEED)

    rate = np.zeros((len(DELAY_LIMITS), len(CG_LIMITS)))
    count = np.zeros_like(rate, dtype=int)
    print("\nPass rate [%] (number of samples) by aft CG limit and maximum delay")
    print("delay \\ CG " + "".join(f"{x:>11.2f}" for x in CG_LIMITS))
    for i, d in enumerate(DELAY_LIMITS):
        row = f"{1000 * d:6.0f} ms  "
        for j, x in enumerate(CG_LIMITS):
            passed, n = pass_rate(results, max_xcg=x, max_delay=d)
            rate[i, j] = 100.0 * passed / n if n else np.nan
            count[i, j] = n
            row += f"{rate[i, j]:6.1f} ({n:3d})"
        print(row)

    # Limits with no failures, preferring the ones that keep the most samples
    clean = [(count[i, j], CG_LIMITS[j], DELAY_LIMITS[i])
             for i in range(len(DELAY_LIMITS)) for j in range(len(CG_LIMITS))
             if rate[i, j] == 100.0]
    print("\nLimits with no failures:")
    for n, x, d in sorted(clean, reverse=True)[:5]:
        bound = 100.0 * failure_rate_upper_bound(n)
        print(f"  CG <= {x:.2f}, delay <= {1000 * d:.0f} ms: {n} samples, "
              f"failure rate < {bound:.1f} % with 95 % confidence")
    if not clean:
        print("  none in this grid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(rate, origin="lower", aspect="auto", cmap="RdYlGn", vmin=80, vmax=100,
                   extent=(CG_LIMITS[0] - 0.005, CG_LIMITS[-1] + 0.005,
                           1000 * DELAY_LIMITS[0] - 5, 1000 * DELAY_LIMITS[-1] + 5))
    for i, d in enumerate(DELAY_LIMITS):
        for j, x in enumerate(CG_LIMITS):
            ax.text(x, 1000 * d, f"{rate[i, j]:.0f}%\n({count[i, j]})",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="samples meeting all requirements [%]")
    ax.set_xlabel("aft CG limit [fraction of chord]")
    ax.set_ylabel("maximum delay [ms]")
    ax.set_title("CG-scheduled pitch SAS: pass rate within CG and delay limits")
    fig.tight_layout()
    fig.savefig(OUT / "requirements.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'requirements.png').resolve()}")


if __name__ == "__main__":
    main()