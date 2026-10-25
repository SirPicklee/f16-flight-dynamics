"""Compare fixed and scheduled SAS gains between the design grid points."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.envelope import survey_point
from f16.linearize import LONG_STATES, linearize, modes, submatrix
from f16.sas import SasGains, closed_loop_matrix
from f16.schedule import build_schedule
from f16.trim import trim_level

OUT = Path("figures")
FIXED = SasGains(k_alpha=0.7121, k_q=0.2007, k_r=0.6619)  # nominal design point


def short_period(a, b, g):
    acl = closed_loop_matrix(a, b, g)
    osc = [m for m in modes(submatrix(acl, LONG_STATES)) if m.is_oscillatory]
    return max(osc, key=lambda m: m.wn) if osc else None


def main():
    OUT.mkdir(exist_ok=True)
    schedule = build_schedule()
    print(f"Gains designed at {schedule.feasible.sum()} grid points")

    # Evaluate at the centre of every grid cell, i.e. away from the design points
    results = {"fixed gains": [], "scheduled gains": []}
    for alt in np.arange(2500.0, 45000.0, 5000.0):
        for vt in np.arange(275.0, 950.0, 50.0):
            if not survey_point(vt, alt).feasible:
                continue
            r = trim_level(vt, alt=alt)
            a, b = linearize(r.x, r.u)
            for name, g in (("fixed gains", FIXED), ("scheduled gains", schedule.gains(vt, alt))):
                sp = short_period(a, b, g)
                if sp is not None:
                    results[name].append((sp.wn, sp.zeta))

    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, pts), color in zip(results.items(), ("tab:gray", "tab:blue")):
        wn, zeta = np.array(pts).T
        ax.scatter(wn, zeta, s=20, color=color, label=name, alpha=0.8)
        print(f"{name:16s} wn {wn.min():.2f} - {wn.max():.2f} rad/s, "
              f"zeta {zeta.min():.2f} - {zeta.max():.2f}  ({len(wn)} points)")
    ax.scatter([3.0], [0.7], marker="x", s=120, color="tab:red", label="design target")
    ax.set_xlabel("short period natural frequency [rad/s]")
    ax.set_ylabel("short period damping ratio")
    ax.set_title("Short period across the envelope, between design points")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gain_schedule.png", dpi=150)
    print(f"Figure saved to {(OUT / 'gain_schedule.png').resolve()}")


if __name__ == "__main__":
    main()