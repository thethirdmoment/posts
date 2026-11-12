"""Checks of the article's model, on seeds the article does not use."""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

SMALL = replace(a.BASE, days=6000, seed=101)


def runs(p):
    market = a.draw_market(p)
    return {k: a.run(p, n, market) for k, n in a.schedules(p).items()}


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_timing_part_averages_zero(seed):
    p = replace(SMALL, seed=seed)
    for r in runs(p).values():
        timing = r["timing"]
        se = timing.std(ddof=1) / math.sqrt(len(timing))
        assert abs(timing.mean()) < 2.5 * se


def test_decomposition_adds_up_every_day():
    for p in (SMALL, replace(SMALL, drift=SMALL.signal)):
        for r in runs(p).values():
            parts = r["parts"]
            fixed = parts["fixed"] + parts["temporary"] + parts["permanent"]
            total = fixed + parts["drift"] + r["timing"]
            assert np.allclose(r["shortfall"], total, rtol=0, atol=1e-4)


def test_simulation_matches_closed_form():
    for r in runs(SMALL).values():
        s = a.summary(r)
        se = s["sd"] / math.sqrt(SMALL.days)
        assert abs(s["mean"] - s["cf_mean"]) < 3 * se
        assert abs(s["sd"] / s["cf_sd"] - 1) < 0.03


def test_flat_profile_gives_almgren_chriss_sinh(monkeypatch):
    p = a.BASE
    flat = np.full(p.slices, p.adv / p.slices)
    monkeypatch.setattr(a, "volume_profile", lambda _: flat)
    tau = 1 / p.slices
    eta = p.eta / p.adv  # temporary impact per unit of trading rate
    eta_tilde = eta - p.gamma * tau / 2

    for hours in (4, 2, 1, 0.5):
        lam = a.urgency(p, hours)
        kappa_tilde2 = lam * p.sigma**2 / eta_tilde
        kappa = math.acosh(tau**2 * kappa_tilde2 / 2 + 1) / tau
        t = np.arange(p.slices + 1) * tau
        held = p.shares * np.sinh(kappa * (1 - t)) / np.sinh(kappa)
        expected = held[:-1] - held[1:]
        assert np.allclose(a.optimal(p, lam), expected, atol=1e-3)


def test_risk_neutral_without_permanent_impact_is_vwap():
    p = replace(a.BASE, gamma=0.0)
    assert np.allclose(a.optimal(p, 0.0), a.vwap(p), atol=1e-6)


def test_risk_neutral_is_close_to_vwap():
    p = a.BASE
    gap = np.abs(a.optimal(p, 0.0) - a.vwap(p)).max()
    assert gap < 0.02 * p.shares / p.slices


def test_schedules_buy_the_whole_order():
    p = a.BASE
    lam = a.urgency(p, 2)
    every = [*a.schedules(p).values(), a.optimal(p, lam, p.signal)]
    for n in every:
        assert n.sum() == pytest.approx(p.shares)


def test_frontier_is_monotone_and_dominates():
    p = a.BASE
    means, sds = a.frontier_curve(p, points=200)
    assert np.all(np.diff(means) > -1e-9)
    assert np.all(np.diff(sds) < 1e-9)
    for n in (a.twap(p), a.vwap(p)):
        parts = a.cost_parts(p, n)
        cost = parts["fixed"] + parts["temporary"] + parts["permanent"]
        sd = float(a.bp(math.sqrt(parts["variance"])))
        if sd <= sds[0]:
            frontier = np.interp(sd, sds[::-1], means[::-1])
            assert float(a.bp(cost)) >= frontier - 1e-6
        else:
            assert float(a.bp(cost)) >= means[0] - 1e-6


def test_vwap_beats_twap_and_front_loading_cuts_risk():
    r = runs(replace(SMALL, seed=404))
    diff, se = a.paired(r["vwap"], r["twap"])
    assert diff > 3 * se
    s = {k: a.summary(x) for k, x in r.items()}
    assert s["vwap"]["sd"] > s["h4"]["sd"] > s["h2"]["sd"] > s["h1"]["sd"]
    assert s["vwap"]["mean"] < s["h2"]["mean"] < s["h1"]["mean"]


def test_vwap_slippage_hides_part_of_the_cost():
    r = runs(replace(SMALL, seed=505))["vwap"]
    slip, shortfall = a.bp(r["slippage"]), a.bp(r["shortfall"])
    assert 0 < slip.mean() < shortfall.mean()
    assert slip.std() < 0.1 * shortfall.std()


def test_same_seed_same_days():
    p = replace(SMALL, days=200)
    one, two = a.draw_market(p), a.draw_market(p)
    for x, y in zip(one, two, strict=True):
        assert np.array_equal(x, y)
    first = a.run(p, a.twap(p), one)["shortfall"]
    again = a.run(p, a.twap(p), two)["shortfall"]
    assert np.array_equal(first, again)
