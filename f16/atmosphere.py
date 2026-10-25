"""
Simplified standard atmosphere and air data computer (ADC).

Based on the ADC routine in Stevens, Lewis & Johnson, Appendix A.
Troposphere model with linear temperature lapse, isothermal above 35,000 ft.
"""

import math

RHO0 = 2.377e-3       # sea-level air density [slug/ft^3]
T0 = 519.0            # sea-level temperature [deg R]
T_STRAT = 390.0       # temperature above 35,000 ft [deg R]
ALT_STRAT = 35000.0   # tropopause altitude [ft]
GAMMA = 1.4           # ratio of specific heats for air
R_AIR = 1716.3        # gas constant for air [ft*lbf/(slug*deg R)]


def atmosphere(alt: float) -> tuple[float, float]:
    """Return (temperature [deg R], density [slug/ft^3]) at altitude [ft]."""
    tfac = 1.0 - 0.703e-5 * alt
    temp = T0 * tfac
    if alt >= ALT_STRAT:
        temp = T_STRAT
    rho = RHO0 * tfac ** 4.14
    return temp, rho


def adc(vt: float, alt: float) -> tuple[float, float]:
    """
    Air data computer.

    vt  : true airspeed [ft/s]
    alt : altitude [ft]
    Returns (mach, qbar), qbar in [lbf/ft^2].
    """
    temp, rho = atmosphere(alt)
    speed_of_sound = math.sqrt(GAMMA * R_AIR * temp)
    mach = vt / speed_of_sound
    qbar = 0.5 * rho * vt ** 2
    return mach, qbar