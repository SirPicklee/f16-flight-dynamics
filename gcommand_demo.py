"""Nonlinear g command response compared with the linear design model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import StateSpace, step

from f16.dynamics import ALPHA, Q
from f16.fcs import DE, FlightControlSystem
from f16.gcommand import design_g_command, nz_command_loop
from f16.linearize import linearize
from f16.trim import trim_level

OUT = Path("figures")
T_STEP, T_FINAL = 0.5, 5.0


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g, ki = design_g_command(a, b, r.x, r.u, wn=3.0, zeta=0.7, integral_pole=3.0)
    print(f"Joint design: k_alpha = {g.k_alpha:.3f}, k_q = {g.k_q:.3f}, ki = {ki:.3f}\n")

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for size in (1.0, 3.0):
        def pilot(t, size=size):
            p = np.zeros(4)
            p[1] = size if t >= T_STEP else 0.0
            return p
        fcs = FlightControlSystem(r.x, r.u, g, g_command=(0.0, ki))
        t, z = fcs.simulate(pilot, T_FINAL)
        dnz = np.array([fcs.nz(zi, pilot(ti)) - fcs.nz0 for ti, zi in zip(t, z)])
        print(f"{size:.0f} g command: nz increment after {T_FINAL - T_STEP:.1f} s = "
              f"{dnz[-1]:.3f} g, peak {dnz.max():.3f} g, "
              f"alpha {np.degrees(z[-1, ALPHA]):.1f} deg, elevator {z[-1, DE]:.1f} deg")
        axes[0].plot(t, dnz / size, label=f"nonlinear, {size:.0f} g")
        axes[1].plot(t, np.degrees(z[:, Q]), label=f"{size:.0f} g")
        axes[2].plot(t, z[:, DE], label=f"{size:.0f} g")

    m, bm, cm = nz_command_loop(a, b, r.x, r.u, g, 0.0, ki)
    tl = np.linspace(0.0, T_FINAL - T_STEP, 2001)
    _, yl = step(StateSpace(m, bm, cm, np.zeros((1, 1))), T=tl)
    axes[0].plot(tl + T_STEP, np.asarray(yl).ravel(), "k--", label="linear design model")

    axes[0].set_ylabel("nz increment / command")
    axes[1].set_ylabel("q [deg/s]")
    axes[2].set_ylabel("elevator [deg]")
    axes[0].set_title("g command, joint SAS + integral design, 502 ft/s, sea level")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "gcommand_nonlinear.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'gcommand_nonlinear.png').resolve()}")


if __name__ == "__main__":
    main()