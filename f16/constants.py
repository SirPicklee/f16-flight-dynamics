"""
F-16 model constants.

Reference: Stevens, Lewis & Johnson, "Aircraft Control and Simulation",
Appendix A (F-16 model based on NASA TP-1538 wind tunnel data).

Units: English system (ft, slug, lbf, s, deg Rankine), matching the
reference so results can be compared directly.
"""

# --- Geometry ---
S = 300.0        # wing reference area [ft^2]
B = 30.0         # wing span [ft]
CBAR = 11.32     # mean aerodynamic chord [ft]
XCGR = 0.35      # reference CG location [fraction of CBAR]

# --- Mass properties ---
G = 32.17        # gravitational acceleration [ft/s^2]
MASS = 1.0 / 1.57e-3  # [slug], reference gives inverse mass 1/m = 1.57e-3 [1/slug]
WEIGHT = MASS * G     # [lbf], about 20,490 lbf

JXX = 9496.0     # roll moment of inertia [slug*ft^2]
JYY = 55814.0    # pitch moment of inertia [slug*ft^2]
JZZ = 63100.0    # yaw moment of inertia [slug*ft^2]
JXZ = 982.0      # product of inertia [slug*ft^2]

# --- Engine ---
HX = 160.0       # engine rotor angular momentum [slug*ft^2/s]

# --- Unit conversions ---
DEG2RAD = 0.017453292519943295
RAD2DEG = 57.29577951308232