"""Tests of the program extracted from the article (article.py).

Every random check uses seeds the article does not use (it uses 2026).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

N = 4000


def test_rule_stops_cuts_and_restores():
    p = a.BASE
    pnl = np.array([0.02, -0.05, -0.08, 0.03])
    peak = np.array([0.0, 0.0, 0.0, 0.03])
    scale = np.array([1.0, 1.0, 1.0, 0.5])
    seated = np.ones(4, bool)
    peak, scale, seated = a.apply_rule(p, pnl, peak, scale, seated)
    assert peak.tolist() == [0.02, 0.0, 0.0, 0.03]
    assert seated.tolist() == [True, True, False, True]
    # the second is cut at a 5% drawdown, the fourth is back at its high
    assert scale.tolist() == [1.0, 0.5, 1.0, 1.0]


def test_no_rule_never_cuts_or_stops():
    p = a.rule("none")
    pnl, peak = np.array([-0.5]), np.array([0.2])
    _, scale, seated = a.apply_rule(p, pnl, peak, np.ones(1), np.ones(1, bool))
    assert scale[0] == 1.0 and seated[0]


def test_losses_carry_forward_until_won_back():
    p = a.BASE
    seated = np.ones(1, bool)
    pay, owed = a.year_end(p, np.array([-0.04]), np.zeros(1), seated)
    assert pay[0] == 0 and owed[0] == pytest.approx(0.04)
    pay, owed = a.year_end(p, np.array([0.03]), owed, seated)
    assert pay[0] == 0 and owed[0] == pytest.approx(0.01)
    pay, owed = a.year_end(p, np.array([0.06]), owed, seated)
    assert pay[0] == pytest.approx(0.2 * 0.05) and owed[0] == 0
    pay, _ = a.year_end(p, np.array([0.06]), np.zeros(1), ~seated)
    assert pay[0] == 0  # let go during the year


@pytest.mark.parametrize("name", list(a.RULES))
def test_no_skill_earns_zero_under_every_rule(name):
    # a fair game stopped by a rule based on its own past stays fair
    p = a.rule(name, replace(a.BASE, ir=0.0))
    c = a.careers(p, n=N, seed=11)
    mean, se = a.mean_se(c["pnl"])
    assert abs(mean) < 3 * se


@pytest.mark.parametrize("ir", [0.0, 0.5, 1.0])
def test_stop_time_matches_the_closed_form(ir):
    # No horizon to speak of. Broadie, Glasserman and Kou shift a barrier
    # checked once a day by 0.5826 sd of a day. A drawdown reads both the
    # high-water mark and the P&L once a day, so we shift by twice that
    # (an approximation for a drawdown stop, which matches to about 1 se).
    p = a.rule("stop only", replace(a.BASE, ir=ir, years=60))
    c = a.careers(p, n=2000, seed=12)
    assert c["tenure"].max() < p.years * a.YEAR
    shift = 2 * 0.5826 * p.vol / math.sqrt(a.YEAR)
    years = a.mean_years_to_stop(ir, p.vol, p.stop + shift)
    mean, se = a.mean_se((c["tenure"] + 1) / a.YEAR)
    assert abs(mean - years) < 4 * se

    peak, peak_se = a.mean_se(c["peak"])
    if ir > 0:
        k = 2 * ir / p.vol
        expected = (math.exp(k * (p.stop + shift)) - 1) / k
        assert abs(peak - expected) < 4 * peak_se


def test_mean_stop_time_values():
    assert a.mean_years_to_stop(0.0, 0.06, 0.075) == pytest.approx(1.5625)
    assert a.mean_years_to_stop(1.0, 0.06, 0.075) == pytest.approx(4.34, abs=0.01)
    # the IR 0 formula is the limit of the general one
    assert a.mean_years_to_stop(1e-6, 0.06, 0.075) == pytest.approx(1.5625, rel=1e-4)


@pytest.mark.parametrize("ir", [0.0, 1.0])
def test_first_year_payout_without_a_rule(ir):
    p = a.rule("none", replace(a.BASE, ir=ir, years=1))
    c = a.careers(p, n=20_000, seed=13)
    pdf = math.exp(-ir**2 / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(ir / math.sqrt(2)))
    expected = p.payout_rate * p.vol * (ir * cdf + pdf)
    mean, se = a.mean_se(c["paid"])
    assert abs(mean - expected) < 3 * se


def test_settings_share_their_shocks():
    p = a.rule("none", replace(a.BASE, ir=0.0))
    c0 = a.careers(p, n=500, seed=14)
    c1 = a.careers(replace(p, ir=1.0), n=500, seed=14)
    drift = 1.0 * p.vol * p.years
    assert np.allclose(c1["pnl"] - c0["pnl"], drift)
    again = a.careers(p, n=500, seed=14)
    assert np.array_equal(again["pnl"], c0["pnl"])


def test_more_skill_keeps_the_job_longer():
    shares = []
    for ir in (0.0, 1.0, 2.0):
        c = a.careers(replace(a.BASE, ir=ir), n=N, seed=15)
        shares.append((c["tenure"] >= 5 * a.YEAR).mean())
    assert shares[0] < shares[1] < shares[2]


def test_the_cut_lengthens_careers():
    p = replace(a.BASE, ir=1.0)
    base = a.careers(p, n=N, seed=16)
    only = a.careers(a.rule("stop only", p), n=N, seed=16)
    assert base["tenure"].mean() > only["tenure"].mean()


SMALL = replace(a.BASE, runs=8, pods=10, years=2)


def test_platform_is_deterministic_and_rules_meet_the_same_hires():
    none = a.platform(a.rule("none", SMALL), seed=17)
    base = a.platform(a.rule("base", SMALL), seed=17)
    assert np.array_equal(none["skill"], base["skill"])
    assert np.array_equal(none["gross"], a.platform(a.rule("none", SMALL), seed=17)["gross"])
    # with no rule nobody is let go, and the seats never empty
    assert none["let_go"].sum() == 0
    assert base["let_go"].sum() > 0


def test_what_the_rules_are_worth_on_another_seed():
    small = replace(a.BASE, runs=100)
    fading = replace(small, fade_years=3.0)
    f = {n: a.fund(a.rule(n, small), seed=19) for n in ("none", "stop only", "base")}
    g = {n: a.fund(a.rule(n, fading), seed=19) for n in ("none", "base")}
    # lasting edge: the stop alone adds, the cut and stop together cost
    d, se = a.paired(f["none"]["net"], f["stop only"]["net"])
    assert d > 3 * se
    d, se = a.paired(f["none"]["net"], f["base"]["net"])
    assert d < -2 * se
    # fading edge: the base rule adds
    d, se = a.paired(g["none"]["net"], g["base"]["net"])
    assert d > 3 * se
    # a steadier fund, and a skilled hire paid less
    assert f["base"]["sharpe"].mean() > f["none"]["sharpe"].mean()
    assert np.nanmean(f["base"]["paid_ir1"]) < np.nanmean(f["none"]["paid_ir1"])


def test_platform_with_no_rule_matches_the_single_manager_payout():
    # with no rule the first hire's payout is the single-manager one
    p = a.rule("none", replace(a.BASE, runs=200, pods=50, years=1, rho=0.0))
    f = a.fund(p, seed=18)
    for ir in (0.0, 1.0):
        pdf = math.exp(-ir**2 / 2) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + math.erf(ir / math.sqrt(2)))
        expected = p.payout_rate * p.vol * (ir * cdf + pdf)
        mean, se = a.mean_se(f[f"paid_ir{ir:.0f}"])
        assert abs(mean - expected) < 3 * se
