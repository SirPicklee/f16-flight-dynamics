"""Compare nonlinear and linear responses to control doublets and save plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, BETA, PHI, P, Q, R, THETA, VT
from f16.linearize import linearize
from f16.simulate import doublet, simulate_linear, simulate_nonlinear
from f16.trim import trim_level

OUT = Path("figures")
XCG = 0.30          # forward CG: open-loop stable, clean comparison
T_FINAL = 20.0


def make_control(u0, channel, amplitude):
    def control(t):
        u = u0.copy()
        u[channel] += doublet(t, start=1.0, width=1.0, amplitude=amplitude)
        return u
    return control


def plot(t, xn, xl, states, labels, title, filename):
    fig, axes = plt.subplots(len(states), 1, figsize=(8, 2.2 * len(states)), sharex=True)
    for ax, i, label in zip(axes, states, labels):
        scale = 1.0 if i == VT else np.degrees(1.0)
        ax.plot(t, xn[:, i] * scale, label="nonlinear", linewidth=2)
        ax.plot(t, xl[:, i] * scale, "--", label="linear", linewidth=1.5)
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
    r = trim_level(502.0, xcg=XCG)
    a, b = linearize(r.x, r.u, xcg=XCG)

    cases = [
        (1, 1.0, (VT, ALPHA, THETA, Q), ("VT [ft/s]", "alpha [deg]", "theta [deg]", "q [deg/s]"),
         "Elevator doublet 1 deg", "elevator_doublet_1deg.png"),
        (1, 5.0, (VT, ALPHA, THETA, Q), ("VT [ft/s]", "alpha [deg]", "theta [deg]", "q [deg/s]"),
         "Elevator doublet 5 deg", "elevator_doublet_5deg.png"),
        (3, 2.0, (BETA, PHI, P, R), ("beta [deg]", "phi [deg]", "p [deg/s]", "r [deg/s]"),
         "Rudder doublet 2 deg", "rudder_doublet_2deg.png"),
    ]
    for channel, amp, states, labels, title, filename in cases:
        control = make_control(r.u, channel, amp)
        t, xn = simulate_nonlinear(r.x, control, T_FINAL, xcg=XCG)
        _, xl = simulate_linear(a, b, r.x, r.u, control, T_FINAL)
        plot(t, xn, xl, states, labels, title, filename)
        key = states[1] if channel == 1 else states[0]
        peak = np.max(np.abs(xn[:, key] - r.x[key]))
        err = np.max(np.abs(xn[:, key] - xl[:, key]))
        print(f"{title:28s} peak = {np.degrees(peak):7.3f} deg, "
              f"max linear error = {np.degrees(err):6.3f} deg ({100 * err / peak:5.1f} %)")
    print(f"Figures saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()