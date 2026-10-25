"""Map trim and open-loop stability over the flight envelope."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from f16.envelope import field, survey

OUT = Path("figures")
SPEEDS = np.arange(250.0, 951.0, 25.0)       # ft/s
ALTITUDES = np.arange(0.0, 45001.0, 2500.0)  # ft


def main():
    OUT.mkdir(exist_ok=True)
    grid = survey(SPEEDS, ALTITUDES)
    feasible = sum(p.feasible for row in grid for p in row)
    print(f"{feasible} of {len(SPEEDS) * len(ALTITUDES)} grid points inside the envelope")

    max_real = field(grid, "max_real_long")
    zeta_dr = field(grid, "dutch_roll_zeta")
    alpha = field(grid, "alpha_deg")
    alt_kft = ALTITUDES / 1000.0

    # Envelope maps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    norm = TwoSlopeNorm(vmin=np.nanmin(max_real), vcenter=0.0, vmax=np.nanmax(max_real))
    m0 = axes[0].pcolormesh(SPEEDS, alt_kft, max_real, cmap="RdBu_r", norm=norm,
                            shading="nearest")
    axes[0].contour(SPEEDS, alt_kft, np.ma.masked_invalid(max_real),
                    levels=[0.0], colors="k", linewidths=1.2)
    fig.colorbar(m0, ax=axes[0], label="largest real part, pitch roots [1/s]")
    axes[0].set_title("Pitch stability (red = divergent, blue = stable), xcg = 0.35")
    m1 = axes[1].pcolormesh(SPEEDS, alt_kft, zeta_dr, cmap="viridis", shading="nearest")
    fig.colorbar(m1, ax=axes[1], label="dutch roll damping ratio")
    axes[1].set_title("Dutch roll damping")
    for ax in axes:
        ax.set_facecolor("0.85")  # grey = outside the envelope
        ax.set_xlabel("true airspeed [ft/s]")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("altitude [kft]")
    fig.tight_layout()
    fig.savefig(OUT / "envelope_stability.png", dpi=150)
    plt.close(fig)

    # Stability collapses onto trim angle of attack
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(alpha.ravel(), max_real.ravel(), c=np.repeat(alt_kft, len(SPEEDS)),
                    cmap="plasma", s=18)
    ax.axhline(0.0, color="k", linewidth=1)
    fig.colorbar(sc, ax=ax, label="altitude [kft]")
    ax.set_xlabel("trim angle of attack [deg]")
    ax.set_ylabel("largest real part, pitch roots [1/s]")
    ax.set_title("Pitch stability is set by trim angle of attack")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "stability_vs_alpha.png", dpi=150)
    plt.close(fig)

    worst = min((p for row in grid for p in row if p.feasible), key=lambda p: p.time_to_double)
    print(f"Fastest divergence: time to double {worst.time_to_double:.2f} s at "
          f"VT = {worst.vt:.0f} ft/s, h = {worst.alt:.0f} ft, alpha = {worst.alpha_deg:.2f} deg")
    print(f"Dutch roll damping range: {np.nanmin(zeta_dr):.3f} - {np.nanmax(zeta_dr):.3f}")
    print(f"Figures saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()