import numpy as np
import pytest

from f16.fcs import XI, FlightControlSystem
from f16.gcommand import design_g_command
from f16.linearize import linearize
from f16.sas import design_pitch_sas
from f16.trim import trim_level


@pytest.fixture(scope="module")
def setup():
    r = trim_level(502.0)
    a, b = linearize(r.x, r.u)
    g_joint, ki = design_g_command(a, b, r.x, r.u)
    return r, a, b, g_joint, ki


def step_pilot(size, start=0.5):
    def pilot(t):
        p = np.zeros(4)
        p[1] = size if t >= start else 0.0
        return p
    return pilot


def test_g_command_starts_in_equilibrium(setup):
    r, _, _, g, ki = setup
    fcs = FlightControlSystem(r.x, r.u, g, g_command=(0.0, ki))
    zd = fcs.derivatives(0.0, fcs.initial_state(), lambda t: np.zeros(4))
    assert np.max(np.abs(zd[[0, 1, 7, 13, XI]])) < 1e-9


def test_nonlinear_aircraft_tracks_a_1g_command(setup):
    # The linear design leaves about 0.02 g after 2.5 s; speed loss and
    # nonlinearity add a few hundredths in the full model
    r, _, _, g, ki = setup
    fcs = FlightControlSystem(r.x, r.u, g, g_command=(0.0, ki))
    pilot = step_pilot(1.0)
    t, z = fcs.simulate(pilot, 3.0)
    dnz = fcs.nz(z[-1], pilot(t[-1])) - fcs.nz0
    assert dnz == pytest.approx(1.0, abs=0.08)


def test_direct_elevator_mode_is_unchanged(setup):
    # Without g_command the pilot input is still an elevator angle
    r, a, b, _, _ = setup
    plain = FlightControlSystem(r.x, r.u, design_pitch_sas(a, b))
    z = plain.initial_state()
    cmd = plain.commands(z, np.array([0.0, 2.0, 0.0, 0.0]))
    assert cmd[0] == pytest.approx(r.u[1] + 2.0)