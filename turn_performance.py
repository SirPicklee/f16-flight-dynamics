"""Sustained turn performance over speed and altitude, and maneuver trims."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f16 import constants as c
from f16.atmosphere import adc
from f16.envelope import MACH_LIMIT
from f16.maneuver_trim import (G_LIMIT, max_sustained_turn_rate,
                               trim_coordinated_turn, trim_pull_up)

OUT = Path("figures")
SPEEDS = np.arange(250.0, 951.0, 50.0)
ALTITUDES = (0.0, 10000.0, 20000.0, 30000.0)


def print_maneuvers():
    print("Coordinated level turns, VT = 502 ft/s, sea level")
    print("  rate   bank   alpha  beta    n    throttle  elevator  aileron  rudder")
    for rate in (0.0, 5.0, 10.0, 15.0, 20.0):
        t = trim_coordinated_turn(502.0, 0.0, rate)
        print(f"  {rate:4.0f}  {t.bank_deg:5.1f}  {t.alpha_deg:5.2f}  {t.beta_deg:5.3f} "
              f"{t.load_factor:5.2f}  {t.u[0]:7.3f}  {t.u[1]:8.3f}  {t.u[2]:7.3f}  {t.u[3]:6.3f}")
    print("\nPull-ups, VT = 502 ft/s, sea level")
    print("  q [deg/s]   n     alpha  throttle  elevator")
    for q in (0.0, 5.0, 10.0, 15.0):
        t = trim_pull_up(502.0, 0.0, q)
        print(f"  {q:8.0f}  {t.load_factor:5.2f}  {t.alpha_deg:6.2f}  {t.u[0]:7.3f}  {t.u[1]:8.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    print_maneuvers()

    print("\nSustained turn rate (this takes about a minute)")
    fig, ax = plt.subplots(figsize=(8, 5))
    for alt in ALTITUDES:
        mach = np.array([adc(v, alt)[0] for v in SPEEDS])
        rates = np.array([max_sustained_turn_rate(v, alt) if m <= MACH_LIMIT else 0.0
                          for v, m in zip(SPEEDS, mach)])
        keep = rates > 0
        ax.plot(mach[keep], rates[keep], "o-", label=f"{alt / 1000:.0f} kft")
        best = int(np.argmax(rates))
        print(f"  {alt:7.0f} ft: best {rates[best]:5.2f} deg/s at Mach {mach[best]:.2f}")

    # Structural limit at sea level: turn rate for n = 9
    v = np.linspace(300.0, 980.0, 100)
    mach_sl = np.array([adc(vi, 0.0)[0] for vi in v])
    w_limit = np.degrees(c.G * np.sqrt(G_LIMIT ** 2 - 1.0) / v)
    ax.plot(mach_sl, w_limit, "k--", linewidth=1, label=f"{G_LIMIT:.0f} g limit, sea level")

    ax.set_xlabel("Mach number")
    ax.set_ylabel("sustained turn rate [deg/s]")
    ax.set_title("Sustained level turn performance (full throttle, alpha <= 25 deg)")
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "turn_performance.png", dpi=150)
    print(f"Figure saved to {(OUT / 'turn_performance.png').resolve()}")


if __name__ == "__main__":
    main()