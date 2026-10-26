"""Can the aircraft roll at 60 deg/s at 260 ft/s?"""

from f16.linearize import linearize
from f16.roll import design_roll_command, roll_mode
from f16.trim import trim_level

AILERON_LIMIT = 21.5   # deg
P_CMD = 60.0           # deg/s

for vt in (260.0, 502.0):
    r = trim_level(vt)
    a, b = linearize(r.x, r.u)
    m = roll_mode(a, b)
    kp, kff = design_roll_command(a, b)
    p_max = abs(m.lda) * AILERON_LIMIT / abs(m.lp)
    print(f"VT = {vt:3.0f} ft/s: Lp = {m.lp:6.3f} 1/s, Lda = {m.lda:7.3f} (deg/s^2)/deg, "
          f"max steady roll rate = {p_max:5.1f} deg/s, "
          f"aileron needed for {P_CMD:.0f} deg/s = {abs(kff) * P_CMD:5.1f} deg")