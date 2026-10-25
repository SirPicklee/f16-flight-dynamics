"""Print trim and modal characteristics for a few flight conditions."""

from f16.linearize import LAT_STATES, LONG_STATES, linearize, modes, submatrix
from f16.trim import trim_level


def describe(mode) -> str:
    lam = mode.eigenvalue
    if mode.is_oscillatory:
        return (f"{lam.real:+.4f} ± {lam.imag:.4f}j   "
                f"wn = {mode.wn:.3f} rad/s, zeta = {mode.zeta:.3f}, "
                f"period = {mode.period:.1f} s")
    kind = "stable" if lam.real < 0 else "UNSTABLE"
    return (f"{lam.real:+.4f}              "
            f"time constant = {abs(mode.time_constant):.2f} s ({kind})")


def report(vt: float, alt: float, xcg: float) -> None:
    r = trim_level(vt, alt=alt, xcg=xcg)
    a, _ = linearize(r.x, r.u, xcg=xcg)
    print(f"\nVT = {vt:.0f} ft/s, h = {alt:.0f} ft, xcg = {xcg:.2f}")
    print(f"  trim: throttle = {r.throttle:.4f}, elevator = {r.elevator_deg:.4f} deg, "
          f"alpha = {r.alpha_deg:.4f} deg")
    print("  longitudinal:")
    for m in modes(submatrix(a, LONG_STATES)):
        print("    " + describe(m))
    print("  lateral-directional:")
    for m in modes(submatrix(a, LAT_STATES)):
        print("    " + describe(m))


if __name__ == "__main__":
    for xcg in (0.30, 0.35, 0.38):
        report(502.0, 0.0, xcg)