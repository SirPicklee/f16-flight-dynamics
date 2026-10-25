import pytest

from f16.atmosphere import adc, atmosphere


def test_sea_level_trim_condition():
    """Reference trim point: sea level, VT = 502 ft/s."""
    mach, qbar = adc(502.0, 0.0)
    assert mach == pytest.approx(0.4495, rel=1e-3)
    assert qbar == pytest.approx(299.5, rel=1e-3)


def test_density_decreases_with_altitude():
    altitudes = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]
    densities = [atmosphere(h)[1] for h in altitudes]
    assert all(a > b for a, b in zip(densities, densities[1:]))


def test_isothermal_above_tropopause():
    t1, _ = atmosphere(36000.0)
    t2, _ = atmosphere(45000.0)
    assert t1 == t2 == 390.0