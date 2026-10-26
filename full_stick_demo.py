"""Full aft stick with and without the envelope limiter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, BETA, VT
from f16.fcs import DE, EnvelopeLimiter, FlightControlSystem
from f16.gcommand import design_g_command
from f16.linearize import linearize
from f16.sas import design_yaw_damper
from f16.trim import trim_level

OUT = Path("figures")
T_STICK, T_FINAL = 0.5, 3.0
BETA0 = 0.5        # initial sideslip disturbance [deg]


def pilot(t):
    p = np.zeros(4)
    p[1] = 8.0 if t >= T_STICK else 0.0     # full aft stick: 9 g total
    return p


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g, ki = design_g_command(a, b, r.x, r.u)
    g.k_r = design_yaw_damper(a, b)

    cases = [("no limiter", None),
             ("limiter, start 20, gain 1", EnvelopeLimiter(alpha_start=20.0, alpha_gain=1.0)),
             ("limiter, start 19, gain 3", EnvelopeLimiter())]

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for name, limiter in cases:
        fcs = FlightControlSystem(r.x, r.u, g, g_command=(0.0, ki), limiter=limiter)
        z0 = fcs.initial_state()
        z0[BETA] = np.radians(BETA0)
        t, z = fcs.simulate(pilot, T_FINAL, z0=z0)
        nz = np.array([fcs.nz(zi, pilot(ti)) for ti, zi in zip(t, z)])
        alpha, beta = np.degrees(z[:, ALPHA]), np.degrees(z[:, BETA])
        print(f"{name:26s}: max alpha {alpha.max():5.1f} deg, max nz {nz.max():4.1f} g, "
              f"max |beta| {np.abs(beta).max():5.2f} deg, final speed {z[-1, VT]:.0f} ft/s")
        axes[0].plot(t, alpha, label=name)
        axes[1].plot(t, nz, label=name)
        axes[2].plot(t, beta, label=name)
        axes[3].plot(t, z[:, DE], label=name)

    axes[0].axhline(25.0, color="tab:red", linestyle="--", linewidth=1, label="25 deg")
    axes[0].set_ylabel("alpha [deg]")
    axes[1].set_ylabel("nz [g]")
    axes[2].set_ylabel("beta [deg]")
    axes[3].set_ylabel("elevator [deg]")
    axes[0].set_title("Full aft stick at 502 ft/s, sea level")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "full_stick.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'full_stick.png').resolve()}")


if __name__ == "__main__":
    main()