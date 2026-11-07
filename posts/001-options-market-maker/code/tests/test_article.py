"""Tests of the program extracted from the article (article.py)."""

from dataclasses import replace

import numpy as np
import pytest

import article as a

SHORT = replace(a.BASE, days=2)


@pytest.mark.parametrize("k", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("vol", [0.1, 0.3])
def test_put_call_parity(k, vol):
    c = a.greeks(100.0, k, 0.2, vol, True)["price"]
    p = a.greeks(100.0, k, 0.2, vol, False)["price"]
    assert c - p == pytest.approx(100.0 - k, abs=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("k", [85.0, 100.0, 115.0])
def test_greeks_match_finite_differences(is_call, k):
    s, t, v, h = 100.0, 45 / 252, 0.22, 1e-4
    g = a.greeks(s, k, t, v, is_call)

    def f(s=s, t=t, v=v):
        return float(a.greeks(s, k, t, v, is_call)["price"])

    assert g["delta"] == pytest.approx((f(s=s + h) - f(s=s - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(s=s + 1e-2) - 2 * f() + f(s=s - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(v=v + h) - f(v=v - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(t=t + h) - f(t=t - h)) / (2 * h), rel=1e-6)


def test_pieces_add_up_to_the_change_in_value():
    run = a.simulate(replace(SHORT, seed=3))
    total = sum(run["pnl"][x] for x in a.PIECES)
    assert np.max(np.abs(total - np.diff(run["equity"]))) < 1e-6


def test_same_seed_same_week_and_seeds_differ():
    x, y, z = (a.simulate(replace(SHORT, seed=s))["equity"] for s in (5, 5, 6))
    np.testing.assert_array_equal(x, y)
    assert not np.array_equal(x, z)


def test_quoting_rules_do_not_change_the_market():
    x = a.simulate(replace(SHORT, seed=2))
    y = a.simulate(replace(SHORT, seed=2, half_width_vol=1.5, hedge_band=500.0))
    np.testing.assert_array_equal(x["spot"], y["spot"])
    np.testing.assert_array_equal(x["level"], y["level"])


def test_no_informed_flow_means_only_uninformed_fills():
    run = a.simulate(replace(SHORT, seed=4, fast_ratio=0.0, vol_informed_ratio=0.0))
    assert {f[4] for f in run["fills"]} == {"uninformed"}


def test_quotes_straddle_theo_and_lean_against_the_book():
    theo, vega = np.array([2.50, 0.80]), np.array([16.9, 9.4])
    bid, ask = a.quotes(a.BASE, theo, vega, 0.0)
    assert np.all(bid < theo) and np.all(ask > theo)
    bid_l, ask_l = a.quotes(a.BASE, theo, vega, 5_000.0)  # long vega: lean down to sell
    assert np.all(ask_l <= ask) and np.all(bid_l <= bid) and np.any(ask_l < ask)


def test_vega_limit_stops_the_side_that_adds_risk():
    theo, vega = np.array([2.0, 3.0]), np.array([15.0, 16.0])
    bid, ask = a.quotes(a.BASE, theo, vega, a.BASE.vega_limit)
    assert np.all(np.isnan(bid)) and not np.any(np.isnan(ask))
    bid, ask = a.quotes(a.BASE, theo, vega, -a.BASE.vega_limit)
    assert np.all(np.isnan(ask)) and not np.any(np.isnan(bid))


def test_fills_sit_on_the_tick_grid():
    cents = np.array([f[3] for f in a.simulate(replace(SHORT, seed=8))["fills"]]) * 100
    np.testing.assert_allclose(cents, np.round(cents), atol=1e-6)


def test_zero_minute_markout_is_each_fills_edge():
    p = replace(a.BASE, days=2, seed=6)
    run = a.simulate(p)
    mk = a.markouts(run, p)
    edges = []
    for m, i, q, price, kind, delta, vega in run["fills"]:
        if m + max(a.HORIZONS) < len(run["spot"]):
            vol = max(run["level"][m] + run["smile"][i], 0.03)
            theo = float(a.greeks(run["spot"][m], run["k"][i], run["tau"][m], vol, run["is_call"][i])["price"])
            edges.append((theo - price) * np.sign(q) * p.multiplier)
    total, count = (sum(mk[kd][0][j] for kd in a.KINDS) for j in (0, 1))
    assert count == len(edges) and total / count == pytest.approx(np.mean(edges), rel=1e-9)


def average(p, seeds=range(40, 46)):
    """The article's averages over six weeks the article itself never uses (it uses seeds 0 to 39)."""
    weeks = [a.week(replace(p, seed=s)) for s in seeds]
    out = {x: np.mean([w["days"][x].mean() for w in weeks]) for x in (*a.PIECES, "total")}
    out["mean_abs_vega"] = np.mean([w["mean_abs_vega"] for w in weeks])
    out["daily_std"] = np.concatenate([w["days"]["total"] for w in weeks]).std(ddof=1)
    return out


def test_the_articles_claims_hold_on_other_seeds():
    calm, toxic = average(a.informed(0.0)), average(a.informed(1.0))
    assert toxic["edge"] > calm["edge"] and toxic["total"] < calm["total"]  # edge up, P&L down
    flat, lean = average(replace(a.BASE, lean_vol=0.0)), average(a.BASE)
    assert lean["mean_abs_vega"] < flat["mean_abs_vega"] and lean["daily_std"] < flat["daily_std"]
    band = average(replace(a.BASE, hedge_band=1000.0))
    assert band["hedging"] > lean["hedging"]  # a band hedges less, so costs less
