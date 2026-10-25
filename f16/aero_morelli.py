"""
Polynomial F-16 aerodynamic model (Morelli, 1998).

A global nonlinear parametric fit to the same NASA TP-1538 wind-tunnel data
as the lookup tables. Being smooth, its derivatives are continuous, so
linearized stability does not jump at table breakpoints.

Reference: E. A. Morelli, "Global Nonlinear Parametric Modeling with
Application to F-16 Aerodynamics", American Control Conference, 1998.
Coefficients as in AeroBenchVVPython (GPL-3.0).

Inputs: angles and surface deflections in degrees, rates in rad/s.
Valid for alpha -10 to 45 deg, beta -30 to 30 deg.
"""

import numpy as np

from f16 import constants as c


def _poly(coeffs, x):
    """c0 + c1*x + c2*x^2 + ..."""
    return sum(ck * x ** k for k, ck in enumerate(coeffs))


def morelli_coefficients(vt, alpha_deg, beta_deg, p, q, r, el, ail, rdr, xcg):
    """Total body-axis coefficients (CX, CY, CZ, Cl, Cm, Cn)."""
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    de, da, dr = np.radians(el), np.radians(ail), np.radians(rdr)
    p_hat = p * c.B / (2.0 * vt)
    q_hat = q * c.CBAR / (2.0 * vt)
    r_hat = r * c.B / (2.0 * vt)

    cx0 = (-1.943367e-2 + 2.136104e-1 * a - 2.903457e-1 * de ** 2 - 3.348641e-3 * de
           - 2.060504e-1 * a * de + 6.988016e-1 * a ** 2 - 9.035381e-1 * a ** 3)
    cxq = _poly([4.833383e-1, 8.644627, 1.131098e1, -7.422961e1, 6.075776e1], a)

    cy0 = -1.145916 * b + 6.016057e-2 * da + 1.642479e-1 * dr
    cyp = _poly([-1.006733e-1, 8.679799e-1, 4.260586, -6.923267], a)
    cyr = _poly([8.071648e-1, 1.189633e-1, 4.177702, -9.162236], a)

    cz0 = (_poly([-1.378278e-1, -4.211369, 4.775187, -1.026225e1, 8.399763], a)
           * (1.0 - b ** 2) - 4.354000e-1 * de)
    czq = _poly([-3.054956e1, -4.132305e1, 3.292788e2, -6.848038e2, 4.080244e2], a)

    cl0 = (-1.05853e-1 * b - 5.776677e-1 * a * b - 1.672435e-2 * a ** 2 * b
           + 1.357256e-1 * b ** 2 + 2.172952e-1 * a * b ** 2 + 3.464156 * a ** 3 * b
           - 2.835451 * a ** 4 * b - 1.098104 * a ** 2 * b ** 2)
    clp = _poly([-4.126806e-1, -1.189974e-1, 1.247721, -7.391132e-1], a)
    clr = _poly([6.250437e-2, 6.067723e-1, -1.101964, 9.100087, -1.192672e1], a)
    clda = (-1.463144e-1 - 4.07391e-2 * a + 3.253159e-2 * b + 4.851209e-1 * a ** 2
            + 2.978850e-1 * a * b - 3.746393e-1 * a ** 2 * b - 3.213068e-1 * a ** 3)
    cldr = (2.635729e-2 - 2.192910e-2 * a - 3.152901e-3 * b - 5.817803e-2 * a * b
            + 4.516159e-1 * a ** 2 * b - 4.928702e-1 * a ** 3 * b - 1.579864e-2 * b ** 2)

    cm0 = (-2.029370e-2 + 4.660702e-2 * a - 6.012308e-1 * de - 8.062977e-2 * a * de
           + 8.320429e-2 * de ** 2 + 5.018538e-1 * a ** 2 * de + 6.378864e-1 * de ** 3
           + 4.226356e-1 * a * de ** 2)
    cmq = _poly([-5.19153, -3.554716, -3.598636e1, 2.247355e2, -4.120991e2, 2.411750e2], a)

    cn0 = (2.993363e-1 * b + 6.594004e-2 * a * b - 2.003125e-1 * b ** 2
           - 6.233977e-2 * a * b ** 2 - 2.107885 * a ** 2 * b + 2.141420 * a ** 2 * b ** 2
           + 8.476901e-1 * a ** 3 * b)
    cnp = _poly([2.677652e-2, -3.298246e-1, 1.926178e-1, 4.013325, -4.404302], a)
    cnr = _poly([-3.698756e-1, -1.167551e-1, -7.641297e-1], a)
    cnda = (-3.348717e-2 + 4.276655e-2 * a + 6.573646e-3 * b + 3.535831e-1 * a * b
            - 1.373308 * a ** 2 * b + 1.237582 * a ** 3 * b + 2.302543e-1 * a ** 2
            - 2.512876e-1 * a ** 3 + 1.588105e-1 * b ** 3 - 5.199526e-1 * a * b ** 3)
    cndr = (-8.115894e-2 - 1.156580e-2 * a + 2.514167e-2 * b + 2.038748e-1 * a * b
            - 3.337476e-1 * a ** 2 * b + 1.004297e-1 * a ** 2)

    cx = cx0 + cxq * q_hat
    cy = cy0 + cyp * p_hat + cyr * r_hat
    cz = cz0 + czq * q_hat
    cl = cl0 + clp * p_hat + clr * r_hat + clda * da + cldr * dr
    cm = cm0 + cmq * q_hat + cz * (c.XCGR - xcg)
    cn = (cn0 + cnp * p_hat + cnr * r_hat + cnda * da + cndr * dr
          - cy * (c.XCGR - xcg) * c.CBAR / c.B)
    return cx, cy, cz, cl, cm, cn