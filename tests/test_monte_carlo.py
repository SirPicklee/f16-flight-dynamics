import numpy as np
import pytest

from f16.fcs import loop_margins, pitch_loop
from f16.linearize import linearize
from f16.monte_carlo import (LeadFilter, Result, Sample, evaluate,
                             failure_rate_upper_bound, pass_rate, run)
from f16.sas import SasGains
from f16.trim import trim_level

NOMINAL_GAINS = SasGains(k_alpha=0.7121, k_q=0.2007)


@pytest.fixture(scope="module")
def nominal_trim():
    return trim_level(502.0)


def nominal_sample(**changes) -> Sample:
    s = Sample(vt=502.0, alt=0.0, xcg=0.35, mq_scale=1.0, mde_scale=1.0,
               jyy_scale=1.0, actuator_bw=20.2, delay=0.0)
    for key, value in changes.items():
        setattr(s, key, value)
    return s


def test_nominal_sample_matches_deterministic_analysis(nominal_trim):
    a, b = linearize(nominal_trim.x, nominal_trim.u)
    reference = loop_margins(*pitch_loop(a, b, NOMINAL_GAINS))
    result = evaluate(nominal_sample(), nominal_trim, NOMINAL_GAINS)
    assert result.passed
    assert result.phase_margin == pytest.approx(reference.phase_margin_deg, abs=0.05)
    assert result.gain_reduction_db == pytest.approx(reference.gain_reduction_db, abs=0.05)


def test_delay_reduces_phase_margin_by_crossover_times_delay(nominal_trim):
    # A pure delay tau removes wc * tau radians of phase at crossover
    a, b = linearize(nominal_trim.x, nominal_trim.u)
    wc = loop_margins(*pitch_loop(a, b, NOMINAL_GAINS)).crossover
    tau = 0.05
    pm0 = evaluate(nominal_sample(), nominal_trim, NOMINAL_GAINS).phase_margin
    pm1 = evaluate(nominal_sample(delay=tau), nominal_trim, NOMINAL_GAINS).phase_margin
    assert pm0 - pm1 == pytest.approx(np.degrees(wc * tau), abs=0.5)


def test_slower_actuator_reduces_phase_margin(nominal_trim):
    fast = evaluate(nominal_sample(actuator_bw=25.0), nominal_trim, NOMINAL_GAINS)
    slow = evaluate(nominal_sample(actuator_bw=15.0), nominal_trim, NOMINAL_GAINS)
    assert slow.phase_margin < fast.phase_margin


def test_same_seed_gives_same_samples():
    first = run(3, lambda vt, alt, xcg: NOMINAL_GAINS, seed=7)
    second = run(3, lambda vt, alt, xcg: NOMINAL_GAINS, seed=7)
    for r1, r2 in zip(first, second):
        assert r1.sample == r2.sample
        assert r1.phase_margin == r2.phase_margin


def make_result(xcg, delay, passed):
    s = nominal_sample(xcg=xcg, delay=delay)
    pm = 60.0 if passed else 30.0   # a 30 deg phase margin fails the 45 deg requirement
    return Result(s, True, 2.5, 0.7, pm, -10.0)


def test_pass_rate_filters_by_cg_and_delay():
    results = [make_result(0.30, 0.01, True),
               make_result(0.34, 0.02, True),
               make_result(0.37, 0.02, False),
               make_result(0.33, 0.045, False)]
    assert pass_rate(results) == (2, 4)
    assert pass_rate(results, max_xcg=0.35) == (2, 3)
    assert pass_rate(results, max_xcg=0.35, max_delay=0.03) == (2, 2)


def test_rule_of_three():
    assert failure_rate_upper_bound(100) == pytest.approx(0.03)
    assert failure_rate_upper_bound(300) == pytest.approx(0.01)


def test_lead_filter_peak_phase_at_center():
    lead = LeadFilter(center=3.0, ratio=2.5)
    w = lead.center
    c = (lead.pole / lead.zero) * (1j * w + lead.zero) / (1j * w + lead.pole)
    assert np.degrees(np.angle(c)) == pytest.approx(lead.max_phase_deg, abs=1e-6)


def test_lead_filter_with_unit_ratio_changes_nothing(nominal_trim):
    plain = evaluate(nominal_sample(delay=0.03), nominal_trim, NOMINAL_GAINS)
    neutral = evaluate(nominal_sample(delay=0.03), nominal_trim, NOMINAL_GAINS,
                       lead=LeadFilter(center=3.0, ratio=1.0))
    assert neutral.phase_margin == pytest.approx(plain.phase_margin, abs=1e-6)
    assert neutral.gain_reduction_db == pytest.approx(plain.gain_reduction_db, abs=1e-6)