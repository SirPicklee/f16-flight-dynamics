"""Compare the lookup-table and polynomial (Morelli) aerodynamic models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import aero_coefficients
from f16.envelope import field, survey
from f16.linearize import LAT_STATES, LONG_STATES, linearize, modes, submatrix
from f16.trim import trim_level

OUT = Path("figures")
MODELS = ("tables", "morelli")
SPEEDS = np.arange(250.0, 951.0, 25.0)
ALTITUDES = np.arange(0.0, 45001.0, 5000.0)


def print_nominal():
    print("Nominal point: sea level, VT = 502 ft/s, xcg = 0.35")
    for model in MODELS:
        r = trim_level(502.0, model=model)
        a, _ = linearize(r.x, r.u, model=model)
        print(f"\n  {model}: throttle = {r.throttle:.4f}, elevator = {r.elevator_deg:.4f} deg, "
              f"alpha = {r.alpha_deg:.4f} deg")
        for states, name in ((LONG_STATES, "long"), (LAT_STATES, "lat ")):
            for m in modes(submatrix(a, states)):
                lam = m.eigenvalue
                print(f"    {name} {lam.real:+8.4f} {lam.imag:+8.4f}j  "
                      f"wn = {m.wn:6.3f}  zeta = {m.zeta:6.3f}")


def plot_cm_curves():
    alpha = np.linspace(-10.0, 45.0, 551)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, style in zip(MODELS, ("-", "--")):
        cm = [aero_coefficients(500.0, a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, model)[4]
              for a in alpha]
        ax.plot(alpha, cm, style, label=model, linewidth=1.8)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("angle of attack [deg]")
    ax.set_ylabel("Cm (elevator 0, no rates)")
    ax.set_title("Pitching moment: tables vs polynomial fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cm_tables_vs_morelli.png", dpi=150)
    plt.close(fig)


def plot_stability_vs_alpha():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, color in zip(MODELS, ("tab:gray", "tab:blue")):
        grid = survey(SPEEDS, ALTITUDES, model=model)
        alpha = field(grid, "alpha_deg").ravel()
        max_real = field(grid, "max_real_long").ravel()
        ax.scatter(alpha, max_real, s=14, color=color, alpha=0.7, label=model)
        n = np.sum(~np.isnan(alpha))
        print(f"  {model:8s}: {n} feasible points, largest pitch root "
              f"{np.nanmax(max_real):.3f} 1/s")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("trim angle of attack [deg]")
    ax.set_ylabel("largest real part, pitch roots [1/s]")
    ax.set_title("Pitch stability across the envelope: tables vs polynomial")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "stability_tables_vs_morelli.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    print_nominal()
    plot_cm_curves()
    print("\nEnvelope survey (about 20 s)")
    plot_stability_vs_alpha()
    print(f"Figures saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()