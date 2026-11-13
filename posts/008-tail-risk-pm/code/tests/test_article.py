"""Checks of the article's model, on seeds the article does not use."""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

SMALL = replace(a.BASE, paths=4000, years=10, seed=101)


def test_merton_reduces_to_black_scholes_without_crashes():
    p = a.BASE
    for k in a.STRIKES:
        bs = a.bs_put(1.0, k, a.MONTH, p.sigma, p.r)
        assert a.merton_put(p, k, crash_rate=0.0) == pytest.approx(bs, abs=1e-14)


def test_put_call_parity_holds_for_merton_prices():
    # the call from the same series, written through the put
    p = a.BASE
    lam, k_bar = p.priced_rate, a.crash_size(p)
    n = np.arange(25)
    weights = a.poisson.pmf(n, lam * (1 + k_bar) * a.MONTH)
    sigma_n = np.sqrt(p.sigma**2 + n * p.jump_sd**2 / a.MONTH)
    r_n = p.r - lam * k_bar + n * math.log(1 + k_bar) / a.MONTH
    for k in a.STRIKES:
        puts = a.bs_put(1.0, k, a.MONTH, sigma_n, r_n)
        calls = puts + 1.0 - k * np.exp(-r_n * a.MONTH)
        call = float(weights @ calls)
        put = a.merton_put(p, k)
        assert call - put == pytest.approx(1 - k * math.exp(-p.r * a.MONTH), abs=1e-12)


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_discounted_payoff_matches_price_under_pricing_measure(seed):
    p = replace(SMALL, seed=seed, paths=20_000)
    draws = a.draw(p)
    q = replace(p, mu=p.r)
    x, _ = a.month_returns(q, draws, p.priced_rate)
    for k in a.STRIKES:
        payoff = math.exp(-p.r * a.MONTH) * np.maximum(k - x.ravel(), 0)
        se = payoff.std(ddof=1) / math.sqrt(payoff.size)
        assert abs(payoff.mean() - a.merton_put(p, k)) < 2.5 * se


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3])
def test_expected_return_is_held_fixed(rate):
    draws = a.draw(replace(SMALL, seed=404))
    x, _ = a.month_returns(SMALL, draws, rate)
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean() - math.exp(SMALL.mu * a.MONTH)) < 3 * se


@pytest.mark.parametrize("w", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("h", [0.0, 1.0])
def test_quadrature_matches_simulation(w, h):
    p = replace(SMALL, seed=505, paths=20_000)
    draws = a.draw(p)
    x, _ = a.month_returns(p, draws, p.crash_rate)
    log_r = np.log(a.portfolio_return(p, x, w, h))
    sim = 12 * log_r.mean(axis=1)
    se = sim.std(ddof=1) / math.sqrt(len(sim))
    exact = a.growth_parts(p, w, h)["growth"]
    assert abs(sim.mean() - exact) < 2.5 * se


def test_decomposition_adds_up_and_tail_vanishes_without_jumps():
    for h in (0.0, 1.0):
        g = a.growth_parts(a.BASE, 1.0, h)
        assert g["average"] - g["drag"] + g["tail"] == pytest.approx(g["growth"], abs=1e-12)
    calm = a.growth_parts(a.BASE, 1.0, 0.0, crash_rate=0.0)
    assert abs(calm["tail"]) < 0.002


def test_premium_rises_with_strike_and_priced_crashes():
    p = a.BASE
    prices = [a.merton_put(p, k) for k in a.STRIKES]
    assert np.all(np.diff(prices) > 0)
    assert a.merton_put(p, 0.9, 0.1) < a.merton_put(p, 0.9, 0.2) < a.merton_put(p, 0.9, 0.3)


def test_hedge_gap_rises_with_crash_rate():
    gaps = [a.growth_gap(a.BASE, 1.0, lam) for lam in np.arange(0, 0.41, 0.05)]
    assert np.all(np.diff(gaps) > 0)


def test_geometric_break_even_below_arithmetic():
    p = a.BASE
    arith = a.break_even(lambda lam: a.hedge_profit(p, lam))
    geo = a.break_even(lambda lam: a.growth_gap(p, 1.0, lam))
    assert p.crash_rate < geo < arith


def test_crashes_never_fall_as_rate_rises():
    draws = a.draw(replace(SMALL, paths=500))
    counts = [a.month_returns(SMALL, draws, lam)[1] for lam in (0.05, 0.1, 0.25, 0.4)]
    for low, high in zip(counts, counts[1:]):
        assert np.all(high >= low)


def test_drawdowns_on_a_known_path():
    log_wealth = np.log(np.array([[1.1, 0.55, 0.88, 1.2, 1.0]]))
    deepest, longest = a.drawdowns(log_wealth)
    assert deepest[0] == pytest.approx(0.5)
    assert longest[0] == 2


def test_same_seed_same_paths():
    p = replace(SMALL, paths=200)
    one, two = a.draw(p), a.draw(p)
    for x, y in zip(one, two, strict=True):
        assert np.array_equal(x, y)
    x, _ = a.month_returns(p, one, p.crash_rate)
    first = a.run(p, x, 1.0, 1.0)
    again = a.run(p, x, 1.0, 1.0)
    assert np.array_equal(first["final"], again["final"])
