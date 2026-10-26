"""Roll rate command: 90 deg/s for one second, with and without yaw damper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import BETA, P, PHI
from f16.fcs import DA, DR, FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_roll_command, roll_mode
from f16.sas import SasGains, design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

OUT = Path("figures")
P_CMD, T_ON, T_OFF, T_FINAL = 90.0, 0.5, 1.5, 4.0


def pilot(t):
    p = np.zeros(4)
    p[2] = P_CMD if T_ON <= t < T_OFF else 0.0
    return p


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    m = roll_mode(a, b)
    kp, kff = design_roll_command(a, b, bandwidth=6.0)
    print(f"Roll mode: Lp = {m.lp:.3f} 1/s, Lda = {m.lda:.3f} (deg/s^2)/deg")
    print(f"Gains: kp = {kp:.4f}, kff = {kff:.4f} deg per deg/s\n")

    pitch = design_pitch_sas(a, b)
    cases = [("yaw damper on", design_yaw_damper(a, b)), ("yaw damper off", 0.0)]

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    t_cmd = np.linspace(0.0, T_FINAL, 401)
    axes[0].plot(t_cmd, [pilot(ti)[2] for ti in t_cmd], "k--", label="command")
    for name, k_r in cases:
        g = SasGains(k_alpha=pitch.k_alpha, k_q=pitch.k_q, k_r=k_r)
        fcs = FlightControlSystem(r.x, r.u, g, roll_command=(kp, kff))
        t, z = fcs.simulate(pilot, T_FINAL)
        p_deg, phi, beta = np.degrees(z[:, P]), np.degrees(z[:, PHI]), np.degrees(z[:, BETA])
        i1 = np.argmin(np.abs(t - (T_ON + 0.5)))
        print(f"{name:15s}: roll rate 0.5 s after command {p_deg[i1]:5.1f} deg/s, "
              f"bank at {T_FINAL:.0f} s {phi[-1]:5.1f} deg, max |beta| {np.abs(beta).max():4.2f} deg, "
              f"max |aileron| {np.abs(z[:, DA]).max():4.1f} deg")
        axes[0].plot(t, p_deg, label=name)
        axes[1].plot(t, phi, label=name)
        axes[2].plot(t, beta, label=name)
        axes[3].plot(t, z[:, DA], label=f"aileron, {name}")
        axes[3].plot(t, z[:, DR], "--", label=f"rudder, {name}")

    axes[0].set_ylabel("roll rate [deg/s]")
    axes[1].set_ylabel("bank angle [deg]")
    axes[2].set_ylabel("sideslip [deg]")
    axes[3].set_ylabel("surface [deg]")
    axes[0].set_title(f"Roll rate command, {P_CMD:.0f} deg/s for 1 s, 502 ft/s, sea level")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "roll_command.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'roll_command.png').resolve()}")


if __name__ == "__main__":
    main()