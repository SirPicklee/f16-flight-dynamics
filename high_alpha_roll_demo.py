"""Rolling at high angle of attack: body-axis roll, stability-axis roll and
stability-axis roll with aileron-rudder interconnect (ARI)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, BETA, PHI
from f16.fcs import DA, DR, FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_ari, design_roll_command
from f16.sas import design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

OUT = Path("figures")
VT0 = 260.0
P_CMD, T_ON, T_OFF, T_FINAL = 60.0, 0.5, 1.5, 4.0


def pilot(t):
    p = np.zeros(4)
    p[2] = P_CMD if T_ON <= t < T_OFF else 0.0
    return p


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(VT0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b)
    g.k_r = design_yaw_damper(a, b)
    roll = design_roll_command(a, b)
    k_ari = design_ari(a, b, r.x[ALPHA], roll[1])
    print(f"Trim at {VT0:.0f} ft/s, sea level: alpha = {r.alpha_deg:.1f} deg")
    print(f"ARI gain: {k_ari:.4f} deg of rudder per deg/s "
          f"({k_ari * P_CMD:.1f} deg of rudder at {P_CMD:.0f} deg/s)")
    print(f"Kinematic estimate for body-axis roll to {P_CMD:.0f} deg bank: "
          f"beta ~ alpha * sin(phi) = {r.alpha_deg * np.sin(np.radians(P_CMD)):.1f} deg\n")

    cases = [("body-axis roll", False, 0.0),
             ("stability-axis roll", True, 0.0),
             ("stability-axis roll + ARI", True, k_ari)]

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for name, stability_axis, ari in cases:
        fcs = FlightControlSystem(r.x, r.u, g, roll_command=roll,
                                  stability_axis=stability_axis, ari_gain=ari)
        t, z = fcs.simulate(pilot, T_FINAL)
        phi, beta, alpha = (np.degrees(z[:, i]) for i in (PHI, BETA, ALPHA))
        print(f"{name:26s}: max |beta| {np.abs(beta).max():5.2f} deg, "
              f"bank at {T_FINAL:.0f} s {phi[-1]:5.1f} deg, "
              f"alpha {alpha.min():4.1f} - {alpha.max():4.1f} deg, "
              f"max |aileron| {np.abs(z[:, DA]).max():4.1f}, "
              f"max |rudder| {np.abs(z[:, DR]).max():4.1f} deg")
        axes[0].plot(t, phi, label=name)
        axes[1].plot(t, beta, label=name)
        axes[2].plot(t, alpha, label=name)
        axes[3].plot(t, z[:, DR], label=name)

    axes[0].set_ylabel("bank [deg]")
    axes[1].set_ylabel("sideslip [deg]")
    axes[2].set_ylabel("alpha [deg]")
    axes[3].set_ylabel("rudder [deg]")
    axes[0].set_title(f"{P_CMD:.0f} deg/s roll for 1 s at alpha = {r.alpha_deg:.0f} deg "
                      f"({VT0:.0f} ft/s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "high_alpha_roll.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'high_alpha_roll.png').resolve()}")


if __name__ == "__main__":
    main()