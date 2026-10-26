"""Monte Carlo robustness of the pitch SAS: fixed, scheduled and CG-scheduled gains."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.monte_carlo import (GAIN_REDUCTION_REQUIRED, PM_REQUIRED,
                             ZETA_SP_RANGE, run)
from f16.sas import SasGains
from f16.schedule import build_cg_schedule, build_schedule

OUT = Path("figures")
N_SAMPLES = 300
SEED = 1
WN = 2.5
FIXED = SasGains(k_alpha=0.7121, k_q=0.2007, k_r=0.6619)


def summarize(name, results):
    passed = sum(r.passed for r in results)
    print(f"\n{name}: {passed} of {len(results)} samples pass "
          f"({100 * passed / len(results):.1f} %)")
    fails = [r for r in results if not r.passed]
    print(f"  failures by requirement (a sample can fail several):")
    print(f"    unstable                  : {sum(not r.stable for r in fails)}")
    print(f"    phase margin < {PM_REQUIRED:.0f} deg     : "
          f"{sum(r.phase_margin < PM_REQUIRED for r in fails)}")
    print(f"    gain margin < 6 dB        : "
          f"{sum(r.gain_reduction_db > GAIN_REDUCTION_REQUIRED for r in fails)}")
    print(f"    short-period damping      : "
          f"{sum(not ZETA_SP_RANGE[0] <= r.sp_zeta <= ZETA_SP_RANGE[1] for r in fails)}")
    pm = np.array([r.phase_margin for r in results])
    print(f"  phase margin: min {pm.min():.1f}, median {np.median(pm):.1f} deg")


def drivers(results):
    """Mean of each uncertainty for passing and failing samples."""
    fields = ("xcg", "delay", "actuator_bw", "mq_scale", "mde_scale", "jyy_scale", "alt")
    ok = [r.sample for r in results if r.passed]
    bad = [r.sample for r in results if not r.passed]
    if not bad:
        print("  no failures")
        return
    print("\n  uncertainty      mean (pass)   mean (fail)")
    for f in fields:
        print(f"  {f:14s} {np.mean([getattr(s, f) for s in ok]):12.3f} "
              f"{np.mean([getattr(s, f) for s in bad]):12.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Running {N_SAMPLES} samples for each controller (about two minutes)...")
    schedule = build_schedule(wn=WN)
    cg_schedule = build_cg_schedule(wn=WN)

    controllers = {
        "Fixed gains": lambda vt, alt, xcg: FIXED,
        "Scheduled gains (speed, altitude)": schedule.gains,
        "Scheduled gains (speed, altitude, CG)": cg_schedule.gains,
    }
    results = {name: run(N_SAMPLES, src, seed=SEED) for name, src in controllers.items()}

    for name, res in results.items():
        summarize(name, res)
    last = list(results)[-1]
    print(f"\nWhat drives the failures of '{last}'?")
    drivers(results[last])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 90, 37)
    for (name, res), color in zip(results.items(), ("tab:gray", "tab:orange", "tab:blue")):
        axes[0].hist([r.phase_margin for r in res], bins=bins, alpha=0.5,
                     color=color, label=name)
    axes[0].axvline(PM_REQUIRED, color="tab:red", linestyle="--", label="requirement")
    axes[0].set_xlabel("phase margin [deg]")
    axes[0].set_ylabel("samples")
    axes[0].set_title(f"Pitch loop phase margin, {N_SAMPLES} samples")
    axes[0].legend(fontsize=8)

    for passed, marker, color, label in ((True, "o", "tab:blue", "pass"),
                                         (False, "x", "tab:red", "fail")):
        pts = [r.sample for r in results[last] if r.passed == passed]
        axes[1].scatter([s.xcg for s in pts], [1000 * s.delay for s in pts],
                        marker=marker, color=color, label=label, s=22)
    axes[1].set_xlabel("CG position [fraction of chord]")
    axes[1].set_ylabel("delay [ms]")
    axes[1].set_title("CG-scheduled gains: pass / fail by CG and delay")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'monte_carlo.png').resolve()}")


if __name__ == "__main__":
    main()