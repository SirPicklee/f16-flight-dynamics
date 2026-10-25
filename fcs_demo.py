"""Stability margins and responses of the SAS with actuators and washout."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, Q
from f16.fcs import DE, FlightControlSystem, lateral_loop, loop_margins, pitch_loop
from f16.linearize import linearize, modes
from f16.sas import EL, design_pitch_sas, design_yaw_damper, sas_control
from f16.simulate import doublet, simulate_closed_loop
from f16.trim import trim_level

OUT = Path("figures")


def print_loop(name, aa, ba, k):
    m = loop_margins(aa, ba, k)
    print(f"\n{name}")
    print(f"  phase margin   = {m.phase_margin_deg:6.1f} deg at {m.crossover:.2f} rad/s")
    print(f"  delay margin   = {1000 * m.delay_margin:6.0f} ms")
    if np.isfinite(m.gain_reduction_db):
        print(f"  gain reduction = {m.gain_reduction_db:6.1f} dB before instability")
    else:
        print("  gain reduction = no limit (open loop stable)")
    for md in modes(aa + ba @ k):
        lam = md.eigenvalue
        print(f"    {lam.real:+9.4f} {lam.imag:+8.4f}j  wn = {md.wn:7.3f}  zeta = {md.zeta:.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    g.k_r = design_yaw_damper(a, b, zeta=0.5)

    print_loop("Pitch loop (alpha + q feedback, 20.2 rad/s actuator)", *pitch_loop(a, b, g))
    print_loop("Yaw damper loop (washout tau = 1 s)", *lateral_loop(a, b, g))

    def pilot_for(amplitude, width):
        def pilot(t):
            p = np.zeros(4)
            p[EL] = doublet(t, 1.0, width, amplitude)
            return p
        return pilot

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for amplitude, width, color in ((1.0, 1.0, "tab:blue"), (10.0, 0.3, "tab:red")):
        pilot = pilot_for(amplitude, width)
        t_i, x_i = simulate_closed_loop(
            r.x, lambda t, x: sas_control(x, r.x, r.u, g, pilot(t)), 6.0)
        t_a, z_a = FlightControlSystem(r.x, r.u, g).simulate(pilot, 6.0)
        label = f"{amplitude:.0f} deg doublet"
        axes[0].plot(t_i, np.degrees(x_i[:, ALPHA]), "--", color=color, alpha=0.6,
                     label=f"{label}, ideal surfaces")
        axes[0].plot(t_a, np.degrees(z_a[:, ALPHA]), color=color, label=f"{label}, actuators")
        axes[1].plot(t_a, np.degrees(z_a[:, Q]), color=color, label=label)
        axes[2].plot(t_a, z_a[:, DE], color=color, label=label)
    axes[0].set_ylabel("alpha [deg]")
    axes[1].set_ylabel("q [deg/s]")
    axes[2].set_ylabel("elevator [deg]")
    axes[0].set_title("Pitch SAS with actuator dynamics and rate limits, xcg = 0.35")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "fcs_pitch.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'fcs_pitch.png').resolve()}")


if __name__ == "__main__":
    main()