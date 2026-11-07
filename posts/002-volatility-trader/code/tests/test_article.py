"""Tests of the program extracted from the article (article.py)."""

import json
import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from scipy.special import ndtr

import article as a

SMALL = replace(a.BASE, paths=400, chunk=400, years=1)
GBM = replace(SMALL, xi=1e-9, lam_p=0.0, lam_q=0.0, theta_q=SMALL.theta_p, dv=0.0)  # constant vol, no premium
S, K, T, VOL = 101.0, 100.0, 21 / 252, 0.18


@pytest.fixture(scope="module")
def paths():
    return a.simulate(SMALL, 0)


@pytest.fixture(scope="module")
def trade(paths):
    log_s, v, _ = paths
    return a.explain(SMALL, log_s, v)


def test_straddle_is_call_plus_put_and_parity():
    sd = VOL * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd**2) / sd
    call = S * ndtr(d1) - K * ndtr(d1 - sd)
    put = K * ndtr(sd - d1) - S * ndtr(-d1)
    assert a.straddle(S, K, T, VOL)["value"] == pytest.approx(call + put, abs=1e-12)
    assert call - put == pytest.approx(S - K, abs=1e-12)  # put-call parity at zero rates


@pytest.mark.parametrize("s", [90.0, 100.0, 101.0, 112.0])
def test_greeks_match_finite_differences(s):
    g = a.straddle(s, K, T, VOL)

    def f(s=s, tau=T, vol=VOL):
        return float(a.straddle(s, K, tau, vol)["value"])

    h, hd = 2e-4 * s, 1e-5 * s
    assert g["delta"] == pytest.approx((f(s=s + hd) - f(s=s - hd)) / (2 * hd), rel=1e-6, abs=1e-8)
    assert g["gamma"] == pytest.approx((f(s=s + h) - 2 * f() + f(s=s - h)) / h**2, rel=1e-4, abs=1e-9)
    assert g["vega"] == pytest.approx((f(vol=VOL + 1e-5) - f(vol=VOL - 1e-5)) / 2e-5, rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(tau=T + 1e-6) - f(tau=T - 1e-6)) / 2e-6, rel=1e-5)


def test_theta_is_minus_half_dollar_gamma_times_variance():
    g = a.straddle(S, K, T, VOL)
    assert g["theta"] == pytest.approx(-0.5 * g["gamma"] * S**2 * VOL**2, rel=1e-12)


def test_index_is_a_martingale(paths):
    st = np.exp(paths[0][:, -1])
    assert abs(st.mean() - SMALL.s0) < 3 * st.std(ddof=1) / math.sqrt(len(st))


def test_closed_form_expected_variance_matches_monte_carlo(paths):
    rv2 = a.realized_vol(SMALL, paths[0])[:, 0] ** 2  # each path's first month
    want = a.expected_variance(SMALL, a.long_run_var(SMALL, SMALL.theta_p, SMALL.lam_p), SMALL.tenor / a.YEAR,
                               SMALL.theta_p, SMALL.lam_p)
    assert abs(rv2.mean() - want) < 3 * rv2.std(ddof=1) / math.sqrt(len(rv2))


def test_long_run_variance_is_stationary(paths):
    assert paths[1].mean() == pytest.approx(a.long_run_var(SMALL, SMALL.theta_p, SMALL.lam_p), rel=0.08)


def test_floored_variance_keeps_the_paths_finite(paths):
    """The Feller condition fails at the article's settings, so variance can dip below zero within a step."""
    assert 2 * SMALL.kappa * SMALL.theta_p < SMALL.xi**2
    assert np.isfinite(paths[0]).all() and np.isfinite(paths[1]).all() and paths[1].min() > -0.01


def test_jump_free_twin_shares_the_brownian_draws():
    p = replace(SMALL, paths=5, chunk=5, years=4, dv=0.0)  # no variance jump, so both worlds start at the same variance
    (ls_a, v_a, crash), (ls_b, v_b, crash_b) = a.simulate(p, 0), a.simulate(a.no_jumps(p), 0)
    assert crash.any() and not crash_b.any()
    np.testing.assert_array_equal(v_a, v_b)
    steps = np.diff(ls_a - ls_b, axis=1).reshape(5, -1, p.substeps)
    calm = ~crash[..., None] | (np.arange(p.substeps) < p.substeps - 1)  # every step but a crash day's close
    assert np.ptp(steps[calm]) < 1e-12  # only the constant crash offset separates the two price paths


def test_same_seed_same_paths_and_chunks_differ():
    p = replace(SMALL, paths=6, chunk=3)
    np.testing.assert_array_equal(a.simulate(p, 0)[0], a.simulate(p, 0)[0])
    assert not np.array_equal(a.simulate(p, 0)[0], a.simulate(p, 1)[0])


def test_atm_vol_sits_below_variance_swap_vol():
    """With stochastic vol and crashes the fair straddle vol is below the variance-swap vol, by under 10%."""
    assert 0.90 < a.atm_ratio(a.BASE, 0.17) < 0.99


def test_atm_vol_equals_variance_swap_vol_at_constant_vol():
    p = a.Params(xi=1e-6, lam_q=0.0, lam_p=0.0)
    assert a.atm_ratio(p, float(a.varswap_vol(p, p.theta_q, 21 / 252))) == pytest.approx(1.0, abs=0.003)


def test_explain_adds_up_per_month(trade):
    m = a.monthly(SMALL, trade)
    oc = -SMALL.option_cost_vol * SMALL.vega_notional
    np.testing.assert_allclose(m["long"], m["gamma_theta"] + m["residual"] + m["hedge_cost"] + oc, atol=1e-9)
    np.testing.assert_allclose(m["short"] + m["long"], 2 * (m["hedge_cost"] + oc), atol=1e-9)


def test_hedged_pnl_is_payoff_less_premium_less_hedge_gains(paths, trade):
    off = a.hedge_grid(SMALL)
    s = a.on_grid(SMALL, np.exp(paths[0]), off)
    k, tau = s[..., 0], (off[-1] - off) / (SMALL.substeps * a.YEAR)
    premium = a.straddle(k, k, tau[0], trade["iv"])["value"]
    delta = a.straddle(s[..., :-1], k[..., None], tau[:-1], trade["iv"][..., None])["delta"]
    direct = trade["units"] * (np.abs(s[..., -1] - k) - premium - (delta * np.diff(s, axis=-1)).sum(-1))
    np.testing.assert_allclose(trade["pnl"].sum(-1), direct, rtol=1e-10, atol=1e-6)


def test_every_straddle_starts_with_the_vega_notional(paths, trade):
    s0 = a.on_grid(SMALL, np.exp(paths[0]), np.array([0]))[..., 0]
    vega = a.straddle(s0, s0, SMALL.tenor / a.YEAR, trade["iv"])["vega"]
    np.testing.assert_allclose(trade["units"] * 0.01 * vega, SMALL.vega_notional)


def test_mark_to_market_sums_to_held_to_expiry(paths, trade):
    mtm = a.mark_to_market(SMALL, paths[0], paths[1], trade)
    np.testing.assert_allclose(mtm.sum(-1), trade["pnl"].sum(-1), atol=1e-6)


def test_at_constant_vol_the_mean_is_zero_and_hedging_more_often_helps():
    log_s, v, _ = a.simulate(GBM, 0)
    stds = []
    for f in (1.0, 4.0, 16.0):
        m = a.monthly(GBM, a.explain(replace(GBM, hedges_per_day=f), log_s, v))
        if f == 1.0:
            per_path = m["pnl"].mean(axis=1)
            assert abs(per_path.mean()) < 3 * per_path.std(ddof=1) / math.sqrt(len(per_path))
        stds.append(m["residual"].std())
    assert stds[0] > stds[1] > stds[2] and stds[2] < 0.25 * stds[0]


def test_hedging_error_is_small_on_calm_days_and_large_on_crash_days(paths, trade):
    he = (trade["pnl"] - trade["gamma_theta"]).reshape(SMALL.paths, -1)
    crash = paths[2][:, : he.shape[1]]
    assert crash.any() and np.abs(he[crash]).mean() > 10 * np.abs(he[~crash]).mean()


def test_hedge_grid():
    assert list(a.hedge_grid(replace(SMALL, hedges_per_day=0.2))) == [0, 80, 160, 240, 320, 336]
    assert len(a.hedge_grid(SMALL)) == SMALL.tenor + 1
    with pytest.raises(ValueError):
        a.hedge_grid(replace(SMALL, hedges_per_day=3.0))


@pytest.fixture(scope="module")
def res():
    return a.results(replace(a.BASE, paths=200, seed=11))[0]  # a seed the article does not use


def test_results_are_deterministic(res):
    again = a.results(replace(a.BASE, paths=200, seed=11))[0]
    assert json.dumps(again, sort_keys=True, default=float) == json.dumps(res, sort_keys=True, default=float)


def test_the_formula_explains_more_than_the_naive_proxy(res):
    e = res["explain"]
    assert e["r2_gamma_theta"] > e["r2_naive"] and e["r2_gamma_theta_no_jump_months"] > 0.9
    assert e["hedge_error_var_share_jump_months_pct"] > 75  # most hedging error comes from the few jump months


def test_hedging_more_often_helps_without_jumps_but_not_with(res):
    fr = {r["hedges_per_day"]: r for r in res["frequency"]}
    no = [fr[f]["he_std_no_jumps"] for f in sorted(fr)]
    assert all(x > y for x, y in pairwise(no))
    assert fr[16.0]["he_std_no_jumps"] < 0.2 * fr[1.0]["he_std_no_jumps"]
    assert fr[16.0]["he_std_jumps"] > 0.6 * fr[1.0]["he_std_jumps"]
    assert fr[16.0]["hedge_cost"] < 2 * fr[1.0]["hedge_cost"]  # costs are negative, so this is more than double


def test_the_premium_and_who_earns_it(res):
    p, s = res["premium"], res["settings"]
    assert p["gap_vol_pts"][0] > 1.0 and p["rv_below_iv_pct"] > 50
    assert p["short"]["win_pct"] > 50 and p["short"]["skew"] < -1 and p["long"]["skew"] > 1
    assert p["short"]["mean"] > 3 * p["short"]["se"] and p["short"]["mean"] < p["short"]["median"]
    fair = s["fair"]["short_pre_cost"]
    assert fair["win_pct"] > 50 and abs(fair["mean"]) < 4 * fair["se"]  # wins most months, mean near zero
    assert abs(fair["mean"]) < 0.1 * s["base"]["short_pre_cost"]["mean"] and s["fair"]["short"]["mean"] < 0


def test_crash_days_take_back_a_material_share(res):
    c = res["crash"]
    assert c["jump_day_pct"] < 0.5 and c["lost_share_of_gains_pct"] > 5
    assert c["median_worst_month_in_avg_months"] > 3
    assert c["rep_cum_held"][-1] == pytest.approx(c["rep_cum_mtm"][-1], abs=1.0)


def test_derived_numbers_are_finite_and_tables_filled(res):
    assert all(np.isfinite(v) for v in res["derived"].values())
    assert all(isinstance(t, str) and t for t in res["tables"].values())
