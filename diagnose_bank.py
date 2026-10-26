"""Where does the bank angle go? Compare the body roll rate with the Euler bank angle."""

import numpy as np

from f16.dynamics import ALPHA, P, PHI, THETA
from f16.fcs import FlightControlSystem
from f16.linearize import linearize
from f16.roll import design_ari, design_roll_command
from f16.sas import design_pitch_sas, design_yaw_damper
from f16.trim import trim_level

VT0 = 260.0


def pilot(t):
    p = np.zeros(4)
    p[2] = 60.0 if 0.5 <= t < 1.5 else 0.0
    return p


r = trim_level(VT0)
a, b = linearize(r.x, r.u)
g = design_pitch_sas(a, b)
g.k_r = design_yaw_damper(a, b)
roll = design_roll_command(a, b)
k_ari = design_ari(a, b, r.x[ALPHA], roll[1])
print(f"roll gains kp = {roll[0]:.3f}, kff = {roll[1]:.3f}; commanded: 60 deg/s for 1 s\n")

for name, stability_axis, ari in (("body-axis", False, 0.0),
                                  ("stability-axis + ARI", True, k_ari)):
    fcs = FlightControlSystem(r.x, r.u, g, roll_command=roll,
                              stability_axis=stability_axis, ari_gain=ari)
    t, z = fcs.simulate(pilot, 4.0)
    p = np.degrees(z[:, P])

    def at(time):
        return p[np.argmin(np.abs(t - time))]

    print(f"{name:22s}: p at 0.75 / 1.0 / 1.4 s = {at(0.75):5.1f} / {at(1.0):5.1f} / "
          f"{at(1.4):5.1f} deg/s, integral of p = {np.trapezoid(p, t):5.1f} deg, "
          f"final bank = {np.degrees(z[-1, PHI]):5.1f} deg, "
          f"final pitch = {np.degrees(z[-1, THETA]):5.1f} deg")