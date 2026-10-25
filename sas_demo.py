"""Open-loop vs SAS responses at the nominal (unstable) CG, and a modal summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, BETA, P, Q, R, THETA
from f16.linearize import LAT_STATES, LONG_STATES, linearize, modes, submatrix
from f16.sas import (EL, RDR, SasGains, closed_loop_matrix, design_pitch_sas,
                     design_yaw_damper, sas_control)
from f16.simulate import doublet, simulate_closed_loop
from f16.trim import trim_level

OUT = Path("figures")
T_FINAL = 15.0


def print_modes(title, a):
    print(f"  {title}")
    for states, name in ((LONG_STATES, "long"), (LAT_STATES, "lat ")):
        for m in modes(submatrix(a, states)):
            lam = m.eigenvalue
            tag = "UNSTABLE" if lam.real > 0 else ""
            print(f"    {name}  {lam.real:+.4f} {lam.imag:+.4f}j  "
                  f"wn = {m.wn:.3f}  zeta = {m.zeta:.3f}  {tag}")


def run(r, gains, channel, amplitude):
    def controller(t, x):
        pilot = np.zeros(4)
        pilot[channel] = doublet(t, 1.0, 1.0, amplitude)
        return sas_control(x, r.x, r.u, gains, pilot)
    return simulate_closed_loop(r.x, controller, T_FINAL)


def plot(results, states, labels, title, filename):
    fig, axes = plt.subplots(len(states), 1, figsize=(8, 2.2 * len(states)), sharex=True)
    for name, (t, x), style in results:
        for ax, i in zip(axes, states):
            ax.plot(t, np.degrees(x[:, i]), style, label=name, linewidth=1.8)
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)                  # nominal CG 0.35
    a, b = linearize(r.x, r.u)

    gains = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    gains.k_r = design_yaw_damper(a, b, zeta=0.5)
    print(f"Gains: k_alpha = {gains.k_alpha:.4f}, k_q = {gains.k_q:.4f}, "
          f"k_r = {gains.k_r:.4f}  (deg per deg or deg/s)")

    print_modes("open loop:", a)
    print_modes("with SAS:", closed_loop_matrix(a, b, gains))

    open_loop = SasGains()
    pitch = [("open loop", run(r, open_loop, EL, 1.0), "-"),
             ("pitch SAS", run(r, gains, EL, 1.0), "--")]
    plot(pitch, (ALPHA, THETA, Q), ("alpha [deg]", "theta [deg]", "q [deg/s]"),
         "Elevator doublet 1 deg, xcg = 0.35", "sas_pitch.png")

    yaw = [("open loop", run(r, open_loop, RDR, 2.0), "-"),
           ("yaw damper", run(r, gains, RDR, 2.0), "--")]
    plot(yaw, (BETA, P, R), ("beta [deg]", "p [deg/s]", "r [deg/s]"),
         "Rudder doublet 2 deg, xcg = 0.35", "sas_yaw.png")
    print(f"Figures saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()