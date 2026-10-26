"""Tune the PI gains of the g command loop on the linear model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import StateSpace, step

from f16.gcommand import nz_command_loop, step_metrics
from f16.linearize import linearize
from f16.sas import design_pitch_sas
from f16.trim import trim_level

OUT = Path("figures")
KP_VALUES = (0.0, 1.0, 2.0, 4.0)          # deg/g
KI_VALUES = (2.0, 5.0, 10.0, 20.0)        # deg/(g s)


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g = design_pitch_sas(a, b, wn=3.0, zeta=0.7)
    print("Nominal point: sea level, 502 ft/s, xcg = 0.35, pitch SAS wn = 3, zeta = 0.7")
    print("Response of nz to a 1 g step command\n")
    print("  kp    ki    rise[s]  overshoot[%]  settle[s]  initial dip[g]")
    for kp in KP_VALUES:
        for ki in KI_VALUES:
            mt = step_metrics(*nz_command_loop(a, b, r.x, r.u, g, kp, ki))
            if not mt.stable:
                print(f"  {kp:3.0f}  {ki:4.0f}   unstable")
                continue
            print(f"  {kp:3.0f}  {ki:4.0f}   {mt.rise_time:6.2f}  {mt.overshoot:10.1f}  "
                  f"{mt.settling_time:9.2f}  {mt.initial_dip:12.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.linspace(0.0, 4.0, 2001)
    for kp, ki in ((0.0, 5.0), (1.0, 5.0), (1.0, 10.0), (2.0, 10.0)):
        m, bm, cm = nz_command_loop(a, b, r.x, r.u, g, kp, ki)
        _, y = step(StateSpace(m, bm, cm, np.zeros((1, 1))), T=t)
        ax.plot(t, np.asarray(y).ravel(), label=f"kp = {kp:.0f}, ki = {ki:.0f}")
    ax.axhline(1.0, color="k", linewidth=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("nz increment [g]")
    ax.set_title("g command loop: response to a 1 g step (linear model)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gcommand_step.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'gcommand_step.png').resolve()}")


if __name__ == "__main__":
    main()