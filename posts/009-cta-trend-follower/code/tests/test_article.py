"""Checks of the trend follower's maths and of the behaviour the article describes.

The model checks run small worlds on seeds the article does not use.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

SMALL = replace(a.BASE, worlds=40, years=10, seed=101)


@pytest.fixture(scope="module")
def base():
    draws = a.draw(SMALL)
    return draws, a.world(SMALL, draws)


def test_block_identity_is_exact():
    z = np.random.default_rng(5).standard_normal((630, 7))
    pnl, identity, move = a.block_pnl(z, 63)
    assert np.max(np.abs(pnl - identity)) < 1e-10
    assert np.allclose(move, z.reshape(10, 63, 7).sum(1))


def test_block_pnl_averages_zero_on_a_random_walk():
    z = np.random.default_rng(6).standard_normal((63 * 4000, 1))
    pnl = a.block_pnl(z, 63)[0].ravel()
    assert abs(pnl.mean()) < 2 * pnl.std() / math.sqrt(len(pnl))


def test_block_pnl_sign_follows_the_regime():
    p = replace(SMALL, worlds=10)
    draws = a.draw(p)
    for share, sign in ((0.01, 1), (0.95, -1)):
        q = replace(p, range_share=share)
        wd = a.world(q, draws)
        pnl = a.identity_check(q, wd)["pnl"]
        assert sign * pnl.mean() > 0


@pytest.mark.parametrize("share", a.RANGE_SHARES)
def test_range_share_matches_the_chain(share):
    p = replace(SMALL, range_share=share, worlds=200, years=20)
    trend = a.trend_chain(p, a.draw(p)["trend"])
    observed = np.mean(trend == a.RANGE)
    assert abs(observed - share) < 0.04


def test_trend_length_and_storm_share():
    p = replace(SMALL, range_share=0.3, worlds=200, years=20)
    draws = a.draw(p)
    trend = a.trend_chain(p, draws["trend"])
    in_trend = trend[:-1] != a.RANGE
    ended = in_trend & (trend[1:] != trend[:-1])
    assert abs(in_trend.sum() / ended.sum() - p.trend_days) < 25
    storm = a.storm_chain(p, draws["storm"])
    assert abs(storm.mean() - a.storm_share(p)) < 0.02


def test_signal_uses_only_past_returns():
    r = np.random.default_rng(1).standard_normal((50, 2, 3))
    s = a.trend_signal(r, 5)
    expected = np.sign(r[10:15].sum(0))
    assert np.array_equal(s[15], expected)
    assert np.all(s[:5] == 0)


def test_looking_ahead_would_be_obvious(base):
    draws, wd = base
    honest = a.sharpe(a.book(a.run(SMALL, wd)))
    cheat = np.sign(wd["r"])
    peek = a.sharpe(a.book(a.run(SMALL, wd, signal=cheat)))
    assert peek.mean() > 10 * honest.mean() > 0


def test_zero_checks_hold_on_other_seeds():
    for seed in (201, 202):
        p = replace(SMALL, seed=seed, trend_sharpe=0.0, half_life=math.inf)
        draws = a.draw(p)
        wd = a.world(p, draws)
        x = a.run(p, wd)
        gross = a.sharpe(a.book(x) + x["cost"].sum(2))
        mean, se = a.mean_se(gross)
        assert abs(mean) < 2.5 * se

        q = replace(SMALL, seed=seed)
        wq = a.world(q, draws)
        ones = np.ones_like(wq["r"])
        long = a.sharpe(a.book(a.run(q, wq, scheme="per_market", signal=ones)))
        mean, se = a.mean_se(long)
        assert abs(mean) < 2.5 * se


def test_volatility_targeting_equalizes_risk(base):
    draws, wd = base
    x = a.run(SMALL, wd, scheme="per_market")
    vol = x["pnl"].std(0).mean(0)
    assert vol.max() / vol.min() < 1.4
    y = a.run(SMALL, wd)
    realized = a.book(y).std(0) * math.sqrt(a.YEAR)
    assert abs(realized.mean() - SMALL.target_vol) < 0.01


def test_regime_split_adds_up(base):
    draws, wd = base
    x = a.run(SMALL, wd)
    label = a.labels(SMALL, wd, SMALL.lookback)
    shares, parts = a.regime_split(SMALL, x, label)
    total = a.YEAR * a.book(x).mean(0)
    assert np.allclose(parts.sum(0), total)
    assert np.allclose(shares.sum(0), 1)


def test_same_seed_same_world_and_shared_returns(base):
    draws, wd = base
    again = a.world(SMALL, a.draw(SMALL))
    assert np.array_equal(wd["r"], again["r"])
    longer = a.trend_signal(wd["r"], 252)
    assert longer.shape == wd["r"].shape


def test_trend_follower_beats_one_market_and_is_convex(base):
    draws, wd = base
    x = a.run(SMALL, wd)
    daily = a.book(x)
    assert a.sharpe(daily).mean() > a.sharpe(x["pnl"][..., 0]).mean()
    market = wd["r"][SMALL.burn_in :, :, :6].mean(2)
    curve = a.smile(SMALL, daily, market)[0]
    mean, se = a.mean_se(curve[:, 0])
    assert mean > 3 * se
