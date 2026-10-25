"""
Nonlinear 6-DoF equations of motion for the F-16.

Flat, non-rotating Earth; rigid body with constant mass. Translational
dynamics are written in wind-axis variables (VT, alpha, beta), rotational
dynamics in body axes (P, Q, R), attitude as Euler angles.

Reference: Stevens & Lewis, Aircraft Control and Simulation, 2nd ed.,
Sections 2.5 and 3.4, Appendix A.

State vector x (13):
    0  vt     true airspeed            [ft/s]
    1  alpha  angle of attack          [rad]
    2  beta   sideslip angle           [rad]
    3  phi    roll angle               [rad]
    4  theta  pitch angle              [rad]
    5  psi    yaw angle                [rad]
    6  p      roll rate                [rad/s]
    7  q      pitch rate               [rad/s]
    8  r      yaw rate                 [rad/s]
    9  north  north position           [ft]
    10 east   east position            [ft]
    11 alt    altitude                 [ft]
    12 power  engine power level       [%]

Control vector u (4):
    0  throttle  [0, 1]
    1  elevator  [deg]
    2  aileron   [deg]
    3  rudder    [deg]
"""

import math

import numpy as np

from f16 import aero_tables as aero
from f16 import constants as c
from f16.atmosphere import adc
from f16.engine import pdot, tgear, thrust

# Index names so the code reads like the equations
VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, NORTH, EAST, ALT, POW = range(13)


def inertia_coefficients(jxx: float, jyy: float, jzz: float, jxz: float) -> tuple:
    """Coefficients c1..c9 of the body-axis moment equations (Stevens & Lewis, Table 2.4-1)."""
    gam = jxx * jzz - jxz ** 2
    c1 = ((jyy - jzz) * jzz - jxz ** 2) / gam
    c2 = (jxx - jyy + jzz) * jxz / gam
    c3 = jzz / gam
    c4 = jxz / gam
    c5 = (jzz - jxx) / jyy
    c6 = jxz / jyy
    c7 = 1.0 / jyy
    c8 = (jxx * (jxx - jyy) + jxz ** 2) / gam
    c9 = jxx / gam
    return c1, c2, c3, c4, c5, c6, c7, c8, c9


C1, C2, C3, C4, C5, C6, C7, C8, C9 = inertia_coefficients(c.JXX, c.JYY, c.JZZ, c.JXZ)


def aero_coefficients(vt, alpha_deg, beta_deg, p, q, r, el, ail, rdr, xcg):
    """Total body-axis force and moment coefficients (CX, CY, CZ, Cl, Cm, Cn)."""
    d_ail = ail / 20.0
    d_rdr = rdr / 30.0

    cx = aero.cx(alpha_deg, el)
    cy = aero.cy(beta_deg, ail, rdr)
    cz = aero.cz(alpha_deg, beta_deg, el)
    cl = (aero.cl(alpha_deg, beta_deg)
          + aero.dlda(alpha_deg, beta_deg) * d_ail
          + aero.dldr(alpha_deg, beta_deg) * d_rdr)
    cm = aero.cm(alpha_deg, el)
    cn = (aero.cn(alpha_deg, beta_deg)
          + aero.dnda(alpha_deg, beta_deg) * d_ail
          + aero.dndr(alpha_deg, beta_deg) * d_rdr)

    # Damping: rates are made non-dimensional with c/2V (pitch) and b/2V (roll, yaw)
    cxq, cyr, cyp, czq, clr, clp, cmq, cnr, cnp = aero.damping(alpha_deg)
    q_hat = c.CBAR * q / (2.0 * vt)
    b2v = c.B / (2.0 * vt)

    cx += q_hat * cxq
    cy += b2v * (cyr * r + cyp * p)
    cz += q_hat * czq
    cl += b2v * (clr * r + clp * p)
    # CG shift: moments are transferred from the reference CG to the actual CG
    cm += q_hat * cmq + cz * (c.XCGR - xcg)
    cn += b2v * (cnr * r + cnp * p) - cy * (c.XCGR - xcg) * c.CBAR / c.B

    return cx, cy, cz, cl, cm, cn


def f16_derivatives(x: np.ndarray, u: np.ndarray, xcg: float = c.XCGR) -> np.ndarray:
    """Return the state derivative dx/dt for state x and control u."""
    vt, alpha, beta = x[VT], x[ALPHA], x[BETA]
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    p, q, r = x[P], x[Q], x[R]
    alt, power = x[ALT], x[POW]
    throttle, el, ail, rdr = u

    xd = np.zeros(13)

    # Air data and engine
    mach, qbar = adc(vt, alt)
    xd[POW] = pdot(power, tgear(throttle))
    t = thrust(power, alt, mach)

    # Aerodynamics
    cx, cy, cz, cl, cm, cn = aero_coefficients(
        vt, alpha * c.RAD2DEG, beta * c.RAD2DEG, p, q, r, el, ail, rdr, xcg)

    # Body-axis velocity components
    cbeta = math.cos(beta)
    uu = vt * math.cos(alpha) * cbeta
    vv = vt * math.sin(beta)
    ww = vt * math.sin(alpha) * cbeta

    sth, cth = math.sin(theta), math.cos(theta)
    sph, cph = math.sin(phi), math.cos(phi)
    spsi, cpsi = math.sin(psi), math.cos(psi)

    qs = qbar * c.S
    inv_m = 1.0 / c.MASS

    # Force equations (body axes)
    udot = r * vv - q * ww - c.G * sth + inv_m * (qs * cx + t)
    vdot = p * ww - r * uu + c.G * cth * sph + inv_m * qs * cy
    wdot = q * uu - p * vv + c.G * cth * cph + inv_m * qs * cz

    # Convert to wind-axis variables
    uw2 = uu * uu + ww * ww
    xd[VT] = (uu * udot + vv * vdot + ww * wdot) / vt
    xd[ALPHA] = (uu * wdot - ww * udot) / uw2
    xd[BETA] = (vt * vdot - vv * xd[VT]) * cbeta / uw2

    # Euler angle kinematics (singular at theta = +/-90 deg)
    xd[PHI] = p + (sth / cth) * (q * sph + r * cph)
    xd[THETA] = q * cph - r * sph
    xd[PSI] = (q * sph + r * cph) / cth

    # Moment equations (body axes), including engine gyroscopic effect HX
    qsb = qs * c.B
    xd[P] = (C2 * p + C1 * r + C4 * c.HX) * q + qsb * (C3 * cl + C4 * cn)
    xd[Q] = (C5 * p - C7 * c.HX) * r + C6 * (r * r - p * p) + qs * c.CBAR * C7 * cm
    xd[R] = (C8 * p - C2 * r + C9 * c.HX) * q + qsb * (C4 * cl + C9 * cn)

    # Navigation: body velocity rotated into the north-east-up frame
    xd[NORTH] = (uu * cth * cpsi
                 + vv * (sph * sth * cpsi - cph * spsi)
                 + ww * (cph * sth * cpsi + sph * spsi))
    xd[EAST] = (uu * cth * spsi
                + vv * (sph * sth * spsi + cph * cpsi)
                + ww * (cph * sth * spsi - sph * cpsi))
    xd[ALT] = uu * sth - vv * sph * cth - ww * cph * cth

    return xd