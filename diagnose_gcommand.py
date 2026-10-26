"""Joint design of the alpha/q feedback and the nz integral gain."""

import numpy as np
from scipy.signal import StateSpace, step

from f16.gcommand import design_g_command, feedforward_gain, nz_command_loop
from f16.linearize import linearize
from f16.sas import design_pitch_sas
from f16.trim import trim_level

T_FINAL = 3.0

r = trim_level(502.0)
a, b = linearize(r.x, r.u)
t = np.linspace(0.0, T_FINAL, 3001)
last_second = t >= T_FINAL - 1.0


def report(name, g, kp, ki):
    kff = feedforward_gain(a, b, r.x, r.u, g)
    m, _, _ = nz_command_loop(a, b, r.x, r.u, g, kp, ki)
    print(f"{name}: k_alpha = {g.k_alpha:.3f}, k_q = {g.k_q:.3f}, kp = {kp:.2f}, "
          f"ki = {ki:.3f}, kff = {kff:.3f}")
    print(f"    closed-loop poles {np.round(np.sort_complex(np.linalg.eigvals(m)), 3)}")
    for ff, ff_name in ((0.0, "no feedforward"), (kff, "feedforward   ")):
        m, bm, cm = nz_command_loop(a, b, r.x, r.u, g, kp, ki, kff=ff, setpoint_weight=0.0)
        _, y = step(StateSpace(m, bm, cm, np.zeros((1, 1))), T=t)
        y = np.asarray(y).ravel()
        rise = t[np.argmax(y >= 0.9)] if np.any(y >= 0.9) else np.nan
        print(f"    {ff_name}: time to 90 % {rise:4.2f} s, peak {y.max():.3f} g, "
              f"error in last second {np.max(np.abs(1.0 - y[last_second])):.3f} g")
    print()


report("hand-picked SAS + PI ", design_pitch_sas(a, b, wn=3.0, zeta=0.7), 1.0, 5.0)
g_joint, ki_joint = design_g_command(a, b, r.x, r.u)
report("joint design (kp = 0)", g_joint, 0.0, ki_joint)