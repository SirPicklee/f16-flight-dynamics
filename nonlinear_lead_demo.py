"""Nonlinear check of the lead filter at a worst-case corner:
aft CG, high dynamic pressure, slow actuator and long delay."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16.dynamics import ALPHA, Q
from f16.fcs import DE, ELEVATOR, Actuator, FlightControlSystem, loop_margins
from f16.linearize import linearize
from f16.monte_carlo import LeadFilter, Sample, perturbed_pitch_loop
from f16.sas import EL, design_pitch_sas
from f16.schedule import best_wn
from f16.simulate import doublet
from f16.trim import trim_level

OUT = Path("figures")
VT, ALT, XCG = 800.0, 10000.0, 0.37
DELAY = 0.045
SLOW_ELEVATOR = Actuator(15.0, ELEVATOR.position_limit, ELEVATOR.rate_limit)
LEAD = LeadFilter(center=6.0, ratio=2.5)


def pilot(t):
    p = np.zeros(4)
    p[EL] = doublet(t, 1.0, 0.5, 1.0)
    return p


def main():
    OUT.mkdir(exist_ok=True)
    r = trim_level(VT, alt=ALT, xcg=XCG)
    a, b = linearize(r.x, r.u, xcg=XCG)
    g = design_pitch_sas(a, b, wn=best_wn(a, b))
    print(f"Condition: VT = {VT:.0f} ft/s, h = {ALT:.0f} ft, xcg = {XCG:.2f}, "
          f"trim alpha = {r.alpha_deg:.2f} deg")
    print(f"Gains: k_alpha = {g.k_alpha:.3f}, k_q = {g.k_q:.3f}\n")

    cases = [("ideal hardware, no lead", None, 0.0, ELEVATOR),
             ("45 ms delay, slow actuator, no lead", None, DELAY, SLOW_ELEVATOR),
             ("45 ms delay, slow actuator, with lead", LEAD, DELAY, SLOW_ELEVATOR)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for name, lead, delay, actuator in cases:
        hardware = Sample(VT, ALT, XCG, 1.0, 1.0, 1.0, actuator.bandwidth, delay)
        m = loop_margins(*perturbed_pitch_loop(a, b, g, hardware, lead))
        fcs = FlightControlSystem(r.x, r.u, g, xcg=XCG, lead=lead, delay=delay,
                                  elevator=actuator)
        t, z = fcs.simulate(pilot, 6.0)
        residual = np.max(np.abs(np.degrees(z[t > 3.0, Q])))
        print(f"  {name:40s} linear PM = {m.phase_margin_deg:5.1f} deg, "
              f"residual |q| after 3 s = {residual:.3f} deg/s")
        axes[0].plot(t, np.degrees(z[:, ALPHA]), label=name)
        axes[1].plot(t, np.degrees(z[:, Q]), label=name)
        axes[2].plot(t, z[:, DE], label=name)

    axes[0].set_ylabel("alpha [deg]")
    axes[1].set_ylabel("q [deg/s]")
    axes[2].set_ylabel("elevator [deg]")
    axes[0].set_title(f"Nonlinear doublet response, VT = {VT:.0f} ft/s, "
                      f"h = {ALT / 1000:.0f} kft, xcg = {XCG:.2f}")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "nonlinear_lead.png", dpi=150)
    print(f"\nFigure saved to {(OUT / 'nonlinear_lead.png').resolve()}")


if __name__ == "__main__":
    main()