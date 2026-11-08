"""Tests of the program extracted from the article (article.py)."""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

SHORT = replace(a.BASE, days=3)
OTHER_SEEDS = range(100, 106)  # months the article does not use


def equity_change_matches(p):
    run = a.simulate(p)
    total = sum(run["days"][x].sum() for x in a.PIECES)
    return run, total


@pytest.mark.parametrize("hedge", a.HEDGES)
@pytest.mark.parametrize("primary", [True, False])
def test_pieces_add_up_to_the_change_in_equity(hedge, primary):
    p = replace(SHORT, hedge=hedge, primary=primary, seed=11)
    run, total = equity_change_matches(p)
    assert total == pytest.approx(run["equity"], abs=1e-4)


def test_same_seed_same_month_and_settings_share_the_market():
    one = a.simulate(replace(SHORT, seed=5))
    two = a.simulate(replace(SHORT, seed=5))
    assert np.array_equal(one["premium"], two["premium"])
    w1 = a.draw(replace(SHORT, seed=5))
    w2 = a.draw(replace(SHORT, seed=5, hedge="basket", fee=0.0))
    assert w1.nav == w2.nav and w1.u == w2.u


def test_weights_sum_to_one_and_fall_with_rank():
    weight, beta, spread = a.basket(a.BASE)
    assert weight.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weight) < 0) and np.all(np.diff(spread) > 0)
    assert beta[0] == pytest.approx(0.8) and beta[-1] == pytest.approx(1.2)


def test_hedge_ratio_matches_regression_on_other_seeds():
    p = replace(a.BASE, days=10)
    x, y = [], []
    for s in OTHER_SEEDS:
        nav, fut = a.market(p, np.random.default_rng(s))
        y.append(np.diff(nav) / nav[:-1])
        x.append(np.diff(fut) / fut[:-1])
    x, y = np.concatenate(x), np.concatenate(y)
    slope = np.cov(x, y)[0, 1] / x.var(ddof=1)
    assert slope == pytest.approx(a.hedge_ratio(p), abs=0.01)


def test_tracking_error_matches_simulated_residual():
    p = replace(a.BASE, days=20)
    res = []
    for s in OTHER_SEEDS:
        nav, fut = a.market(p, np.random.default_rng(s))
        r_fund = np.diff(nav) / nav[:-1]
        r_fut = np.diff(fut) / fut[:-1]
        res.append(r_fund - a.hedge_ratio(p) * r_fut)
    yearly = np.concatenate(res).std() / math.sqrt(a.DT)
    assert yearly == pytest.approx(a.tracking_error(p), rel=0.03)


def test_quotes_straddle_fair_value_on_cents_and_lean_against_inventory():
    p = a.BASE
    bid, ask = a.quotes(p, 100.004, 0.0)
    assert bid < 100.004 < ask
    assert round(bid * 100) == pytest.approx(bid * 100, abs=1e-9)
    short_bid, short_ask = a.quotes(p, 100.004, -3 * p.unit)
    long_bid, long_ask = a.quotes(p, 100.004, 3 * p.unit)
    assert short_bid > bid and short_ask > ask
    assert long_bid < bid and long_ask < ask


def test_arbitrageurs_trade_only_outside_the_cost_of_creating():
    p = replace(a.BASE, arb_rate=1.0)
    w = a.draw(replace(SHORT, seed=1))
    w = replace(w, u=[[0.0] * 10, [0.0] * 10, [0.0] * 10])
    cost = a.creation_cost_bp(p, 100.0, 3.0) * a.BP * 100
    inside = a.arbitrage_orders(p, w, 0, 99.99, 100.0 + cost / 2, 100.0, 3.0)
    assert inside == []
    rich_bid = 100.0 + 2 * cost
    sold = a.arbitrage_orders(p, w, 0, rich_bid, rich_bid + 0.02, 100.0, 3.0)
    assert sold == [(p.arb_size, rich_bid)]
    off = replace(p, primary=False)
    assert a.arbitrage_orders(off, w, 0, rich_bid, 101, 100.0, 3.0) == []


def test_creation_from_a_basket_hedge_pays_only_the_fee():
    p = replace(a.BASE, hedge="basket")
    desk = a.Desk(0.9, 3.0, inventory=-120_000.0, basket=120_000.0)
    cost, shares = a.create_or_redeem(p, desk, 100.0)
    assert shares == 100_000 and cost == pytest.approx(2 * p.fee)
    assert desk.inventory == -20_000 and desk.basket == 20_000

    desk = a.Desk(0.9, 3.0, inventory=120_000.0)
    cost, shares = a.create_or_redeem(replace(a.BASE), desk, 100.0)
    spread = 100_000 * 100.0 * 3.0 * a.BP
    assert cost == pytest.approx(2 * a.BASE.fee + spread)
    assert desk.inventory == 20_000 and desk.cash == pytest.approx(1e7 - cost)


def test_no_creation_below_the_threshold_or_when_switched_off():
    desk = a.Desk(0.9, 3.0, inventory=-90_000.0)
    p = replace(a.BASE, threshold=2)
    assert a.create_or_redeem(p, desk, 100.0) == (0.0, 0)
    off = replace(a.BASE, primary=False)
    desk = a.Desk(0.9, 3.0, inventory=-500_000.0)
    assert a.create_or_redeem(off, desk, 100.0) == (0.0, 0)


def months(p, seeds=OTHER_SEEDS):
    return [a.simulate(replace(p, seed=s)) for s in seeds]


def test_futures_cost_less_but_swing_more_than_basket_on_other_seeds():
    fut = months(a.BASE)
    bask = months(replace(a.BASE, hedge="basket"))

    def cost(ms):
        return -np.mean([m["days"]["hedging"] + m["days"]["primary"] for m in ms])

    def swing(ms):
        return np.concatenate([m["days"]["total"] for m in ms]).std()

    assert cost(fut) < cost(bask)
    assert swing(fut) > 3 * swing(bask)


def test_switching_off_creation_widens_the_premium_on_other_seeds():
    on = months(a.BASE)
    off = months(replace(a.BASE, primary=False))

    def p99(ms):
        return np.quantile(np.abs(np.concatenate([m["premium"] for m in ms])), 0.99)

    def biggest(ms):
        return max(np.abs(m["inventory"]).max() for m in ms)

    assert p99(off) > 1.5 * p99(on)
    assert biggest(off) > biggest(on)
    assert p99(on) < 2 * a.band_bp(a.BASE)


def test_basket_hedged_tracking_pnl_is_small():
    ms = months(replace(a.BASE, hedge="basket"))
    tracking = np.concatenate([m["days"]["tracking"] for m in ms])
    unhedged = np.concatenate([m["days"]["inventory"] for m in ms])
    assert tracking.std() < 0.1 * unhedged.std()
