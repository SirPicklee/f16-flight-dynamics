"""
F100 engine model: throttle gearing, power lag and thrust tables.

Data: Stevens & Lewis, Aircraft Control and Simulation, 2nd ed., Appendix A,
following the Python port in AeroBenchVVPython (GPL-3.0).

Power level is in percent: 0-50 covers idle to military power,
50-100 covers military to maximum afterburner.
"""

import numpy as np

from f16.aero_tables import interp2

ALT_GRID = np.arange(0.0, 60000.0, 10000.0)  # 0 ... 50,000 ft (6 points)
MACH_GRID = np.arange(0.0, 1.2, 0.2)         # 0 ... 1.0 (6 points)

# Thrust tables [lbf]. Rows: Mach, columns: altitude (transposed to [alt, mach])
THRUST_IDLE = np.array([
    [1060, 670, 880, 1140, 1500, 1860],
    [635, 425, 690, 1010, 1330, 1700],
    [60, 25, 345, 755, 1130, 1525],
    [-1020, -170, -300, 350, 910, 1360],
    [-2700, -1900, -1300, -247, 600, 1100],
    [-3600, -1400, -595, -342, -200, 700],
], dtype=float).T

THRUST_MIL = np.array([
    [12680, 9150, 6200, 3950, 2450, 1400],
    [12680, 9150, 6313, 4040, 2470, 1400],
    [12610, 9312, 6610, 4290, 2600, 1560],
    [12640, 9839, 7090, 4660, 2840, 1660],
    [12390, 10176, 7750, 5320, 3250, 1930],
    [11680, 9848, 8050, 6100, 3800, 2310],
], dtype=float).T

THRUST_MAX = np.array([
    [20000, 15000, 10800, 7000, 4000, 2500],
    [21420, 15700, 11225, 7323, 4435, 2600],
    [22700, 16860, 12250, 8154, 5000, 2835],
    [24240, 18910, 13760, 9285, 5700, 3215],
    [26070, 21075, 15975, 11115, 6860, 3950],
    [28886, 23319, 18300, 13484, 8642, 5057],
], dtype=float).T


def tgear(throttle: float) -> float:
    """Map throttle position [0, 1] to commanded power level [%]."""
    if throttle <= 0.77:
        return 64.94 * throttle
    return 217.38 * throttle - 117.38


def rtau(dp: float) -> float:
    """Inverse time constant [1/s] as a function of power change [%]."""
    if dp <= 25.0:
        return 1.0
    if dp >= 50.0:
        return 0.1
    return 1.9 - 0.036 * dp


def pdot(power: float, cpow: float) -> float:
    """Rate of change of actual power level [%/s].

    power : current power level [%]
    cpow  : commanded power level [%]
    """
    if cpow >= 50.0:
        if power >= 50.0:
            target, rate = cpow, 5.0
        else:
            target = 60.0
            rate = rtau(target - power)
    else:
        if power >= 50.0:
            target, rate = 40.0, 5.0
        else:
            target = cpow
            rate = rtau(target - power)
    return rate * (target - power)


def thrust(power: float, alt: float, mach: float) -> float:
    """Engine thrust [lbf] for power level [%], altitude [ft] and Mach."""
    alt = max(alt, 0.0)
    t_mil = interp2(ALT_GRID, MACH_GRID, THRUST_MIL, alt, mach)
    if power < 50.0:
        t_idle = interp2(ALT_GRID, MACH_GRID, THRUST_IDLE, alt, mach)
        return t_idle + (t_mil - t_idle) * power * 0.02
    t_max = interp2(ALT_GRID, MACH_GRID, THRUST_MAX, alt, mach)
    return t_mil + (t_max - t_mil) * (power - 50.0) * 0.02