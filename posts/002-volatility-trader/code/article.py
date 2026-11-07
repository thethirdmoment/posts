# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import skew

YEAR = 252  # trading days in a year


def straddle(s, k, tau, vol):
    """Black-Scholes value and Greeks of a straddle, at zero rates.

    A straddle is a call plus a put with the same strike and expiry.
    Vega is per 1.00 of vol and theta is per year.
    """
    sd = vol * np.sqrt(tau)
    d1 = (np.log(s / k) + 0.5 * sd**2) / sd
    pdf = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

    return {
        "value": s * (2 * ndtr(d1) - 1) - k * (2 * ndtr(d1 - sd) - 1),
        "delta": 2 * ndtr(d1) - 1,
        "gamma": 2 * pdf / (s * sd),
        "vega": 2 * s * pdf * np.sqrt(tau),
        # equals -0.5 * gamma * s**2 * vol**2
        "theta": -s * pdf * vol / np.sqrt(tau),
    }


def simulate(p, chunk):
    """Log index and variance at every step of one chunk of paths.

    Returns (paths, steps + 1) arrays and each day's crash flag.
    """
    z1, z2, crash, size = draws(p, chunk)
    m, dt = p.substeps, 1 / (YEAR * p.substeps)
    drift = crash_drift(p, p.lam_p)

    log_s = np.empty((len(z1) + 1, z1.shape[1]))
    v = np.empty_like(log_s)
    log_s[0] = math.log(p.s0)
    v[0] = long_run_var(p, p.theta_p, p.lam_p)

    for i in range(len(z1)):
        vp = np.maximum(v[i], 0.0)
        root = np.sqrt(vp * dt)
        log_s[i + 1] = log_s[i] - 0.5 * vp * dt + root * z1[i] - drift
        pull = p.kappa * (p.theta_p - vp) * dt
        v[i + 1] = v[i] + pull + p.xi * root * z2[i]

        if (i + 1) % m == 0:  # the day's close
            day = (i + 1) // m - 1
            log_s[i + 1] += size[day]
            v[i + 1] += p.dv * crash[day]

    return log_s.T, v.T, crash.T


def varswap_vol(p, v, tau):
    """The fair variance-swap strike in vol units, priced in Q."""
    return np.sqrt(expected_variance(p, v, tau, p.theta_q, p.lam_q))


def implied_vol(p, v, tau):
    """The option market's vol for a straddle with tau years left."""
    month = varswap_vol(p, v, p.tenor / YEAR)
    return varswap_vol(p, v, tau) * atm_ratio(p, month)


def explain(p, log_s, v):
    """The long straddle on every month of every path, in dollars.

    Each straddle is traded, hedged and marked at the implied vol of
    the day it opens. The P&L is split per hedge interval.
    """
    off = hedge_grid(p)
    s = on_grid(p, np.exp(log_s), off)
    k = s[..., 0]  # struck at the money
    tau = (off[-1] - off) / (p.substeps * YEAR)  # years left
    v_open = on_grid(p, v, off[:1])[..., 0]
    iv = implied_vol(p, v_open, p.tenor / YEAR)

    # sized so that every straddle starts with the same vega
    g = straddle(s[..., :-1], k[..., None], tau[:-1], iv[..., None])
    vega_point = 0.01 * g["vega"][..., 0]
    units = p.vega_notional / vega_point
    pnl, gamma_theta = hedged_pnl(s, k, tau, iv, g, units)
    return {
        "pnl": pnl,
        "gamma_theta": gamma_theta,
        "hedge_cost": hedge_costs(p, s, g, units),
        "iv": iv,
        "units": units,
    }


def hedged_pnl(s, k, tau, iv, g, units):
    """Hedged P&L per interval, and the gamma-theta P&L."""
    payoff = np.abs(s[..., -1:] - k[..., None])
    value = np.concatenate([g["value"], payoff], axis=-1)
    u, s0, ds = units[..., None], s[..., :-1], np.diff(s, axis=-1)
    pnl = u * (np.diff(value, axis=-1) - g["delta"] * ds)

    # squared return less implied variance times the interval's length
    gap = (ds / s0) ** 2 - iv[..., None] ** 2 * -np.diff(tau)
    gamma_theta = u * 0.5 * g["gamma"] * s0**2 * gap
    return pnl, gamma_theta


def monthly(p, ex):
    """Each straddle-month's totals, for the long and the short side."""
    pnl = ex["pnl"].sum(-1)
    gamma_theta = ex["gamma_theta"].sum(-1)
    hedge_cost = ex["hedge_cost"].sum(-1)
    option_cost = -p.option_cost_vol * p.vega_notional

    return {
        "pnl": pnl,
        "gamma_theta": gamma_theta,
        "residual": pnl - gamma_theta,
        "hedge_cost": hedge_cost,
        "iv": ex["iv"],
        "long": pnl + hedge_cost + option_cost,
        "short": -pnl + hedge_cost + option_cost,
    }


@dataclass(frozen=True)
class Params:
    # the index and its variance in the real world (P)
    s0: float = 100.0
    kappa: float = 3.0  # pull back to the long-run level, per year
    theta_p: float = 0.024  # long-run variance before crashes
    xi: float = 0.40  # vol of variance
    rho: float = -0.70  # correlation of index and variance shocks

    # crashes in the real world
    lam_p: float = 0.25  # crashes per year
    mu_j: float = -0.07  # mean log return on a crash day
    sig_j: float = 0.03  # its standard deviation
    dv: float = 0.03  # variance added on a crash day

    # the pricing world (Q) of the option market
    theta_q: float = 0.046  # long-run variance before crashes
    lam_q: float = 0.70  # crashes per year

    # the trade and its costs
    tenor: int = 21  # trading days in each straddle's life
    vega_notional: float = 10_000.0  # dollars per vol point at entry
    hedges_per_day: float = 1.0  # 0.2 is once every five days
    option_cost_vol: float = 0.25  # vol points of vega per trade
    hedge_cost_bp: float = 0.5  # basis points of index traded

    # the experiment
    years: int = 5  # length of each path
    substeps: int = 16  # simulation steps per trading day
    paths: int = 1000
    chunk: int = 100  # paths simulated at a time
    seed: int = 2026
    pricing_seed: int = 99  # draws for fitting the fair vol


BASE = Params()


def fair(p):
    """The same market with no premium in option prices (Q equal P)."""
    return replace(p, theta_q=p.theta_p, lam_q=p.lam_p)


def no_jumps(p):
    """The jump-free twin of a market, with every crash switched off."""
    return replace(p, lam_p=0.0, lam_q=0.0)


def n_chunks(p):
    """How many chunks of paths the experiment simulates."""
    return -(-p.paths // p.chunk)  # rounded up


def crash_chance(lam):
    """The chance of a crash on any one day, at lam crashes a year."""
    return 1 - np.exp(-lam / YEAR)


def crash_drift(p, lam):
    """The drift per step that offsets the expected crash loss."""
    jump_mean = np.exp(p.mu_j + 0.5 * p.sig_j**2) - 1
    return np.log1p(crash_chance(lam) * jump_mean) / p.substeps


def draws(p, chunk):
    """Every random number one chunk of paths uses, in a fixed order."""
    seeds = np.random.SeedSequence(p.seed).spawn(n_chunks(p))
    rng = np.random.default_rng(seeds[chunk])
    n = min(p.chunk, p.paths - chunk * p.chunk)
    days = p.years * YEAR
    steps = days * p.substeps

    # index shocks, and variance shocks correlated with them
    z1 = rng.standard_normal((steps, n))
    other = rng.standard_normal((steps, n))
    z2 = p.rho * z1 + math.sqrt(1 - p.rho**2) * other

    # at most one crash a day, at the close
    crash = rng.random((days, n)) < crash_chance(p.lam_p)
    jump_sizes = p.mu_j + p.sig_j * rng.standard_normal((days, n))
    size = np.where(crash, jump_sizes, 0.0)
    return z1, z2, crash, size


def long_run_var(p, theta, lam):
    """Long-run mean of variance when each crash adds dv to it."""
    return theta + lam * p.dv / p.kappa


def expected_variance(p, v, tau, theta, lam):
    """Expected annualized variance over the next tau years.

    v is today's variance. The last term adds the squared jumps.
    """
    level = long_run_var(p, theta, lam)
    kt = p.kappa * np.maximum(tau, 1e-12)
    # the average weight on today's variance over the period
    w = np.where(kt > 1e-8, -np.expm1(-kt) / kt, 1.0)
    jumps = lam * (p.mu_j**2 + p.sig_j**2)
    return level + (np.maximum(v, 0.0) - level) * w + jumps


def q_day(p, rng, log_s, v, drift):
    """Move log index and variance through one day in Q, in place."""
    dt = 1 / (YEAR * p.substeps)

    for _ in range(p.substeps):
        z1 = rng.standard_normal(len(v))
        other = rng.standard_normal(len(v))
        z2 = p.rho * z1 + math.sqrt(1 - p.rho**2) * other
        vp = np.maximum(v, 0.0)
        log_s += -0.5 * vp * dt + np.sqrt(vp * dt) * z1 - drift
        v += (
            p.kappa * (p.theta_q - vp) * dt
            + p.xi * np.sqrt(vp * dt) * z2
        )

    jump = rng.random(len(v)) < crash_chance(p.lam_q)
    sizes = p.mu_j + p.sig_j * rng.standard_normal(len(v))
    log_s += np.where(jump, sizes, 0.0)
    v += p.dv * jump


def straddles_in_q(p, n):
    """n hedged one-month straddles struck at 1, simulated in Q.

    Returns each one's variance-swap vol and its payoff less the
    gains of a daily delta hedge.
    """
    rng = np.random.default_rng(p.pricing_seed)  # not the trading draws
    v = rng.uniform(0.05, 0.50, n) ** 2
    x = varswap_vol(p, v, p.tenor / YEAR)
    drift = crash_drift(p, p.lam_q)

    log_s, hedge, s_prev = np.zeros(n), np.zeros(n), np.ones(n)
    for d in range(p.tenor):
        tau = (p.tenor - d) / YEAR
        delta = straddle(s_prev, 1.0, tau, x)["delta"]
        q_day(p, rng, log_s, v, drift)
        s = np.exp(log_s)
        hedge += delta * (s - s_prev)
        s_prev = s

    return x, np.abs(s_prev - 1.0) - hedge


def fit_ratio(x, price, t, bins):
    """Cubic fit of fair straddle vol over variance-swap vol x."""
    xb, ratio, weight = [], [], []
    for idx in np.array_split(np.argsort(x), bins):
        mean = price[idx].mean()
        se = price[idx].std(ddof=1) / math.sqrt(len(idx))
        # an at-the-money straddle is worth 2(2N(vol√t/2) − 1)
        vol = 2 * ndtri(0.5 + 0.25 * mean) / math.sqrt(t)
        xb.append(x[idx].mean())
        ratio.append(vol / xb[-1])
        weight.append(mean / (se * ratio[-1]))  # 1 / its standard error

    xb, ratio, weight = np.array(xb), np.array(ratio), np.array(weight)
    basis = np.vstack([xb**i for i in range(4)]).T * weight[:, None]
    return np.linalg.lstsq(basis, ratio * weight, rcond=None)[0]


_curves = {}  # one fitted curve per pricing world


def atm_ratio(p, x, n=400_000, bins=80):
    """Fair at-the-money straddle vol over variance-swap vol x."""
    key = (p.kappa, p.xi, p.rho, p.mu_j, p.sig_j, p.dv, p.theta_q)
    key += (p.lam_q, p.substeps, p.tenor, p.pricing_seed)

    if key not in _curves:
        x_fit, price = straddles_in_q(p, n)
        _curves[key] = fit_ratio(x_fit, price, p.tenor / YEAR, bins)

    x = np.clip(x, 0.05, 0.50)
    return sum(c * x**i for i, c in enumerate(_curves[key]))


def hedge_grid(p):
    """Step offsets of the hedge times in one straddle's life."""
    step = p.substeps / p.hedges_per_day
    if step < 1 or step != int(step):
        msg = f"{p.hedges_per_day} hedges a day do not fit the steps"
        raise ValueError(msg)

    life = p.tenor * p.substeps
    times = np.arange(0, life, int(step))
    return np.unique(np.append(times, life))  # ends at expiry


def on_grid(p, x, offsets):
    """x at each straddle's hedge times, as (paths, months, offsets)."""
    starts = np.arange(p.years * YEAR // p.tenor) * p.tenor * p.substeps
    return x[:, starts[:, None] + offsets]


def hedge_costs(p, s, g, units):
    """The cost of every hedge trade, from the open to the unwind."""
    zero = np.zeros_like(s[..., :1])
    held = np.concatenate([zero, g["delta"], zero], axis=-1)
    traded = np.abs(np.diff(held, axis=-1))
    cost = -p.hedge_cost_bp * 1e-4 * s * traded * units[..., None]

    # the unwind lands in the last interval
    cost[..., -2] += cost[..., -1]
    return cost[..., :-1]


def mark_to_market(p, log_s, v, ex):
    """The long straddle's hedged P&L per interval, marked each day
    at that day's implied vol. The hedge itself is unchanged."""
    off = hedge_grid(p)
    s = on_grid(p, np.exp(log_s), off)
    var = on_grid(p, v, off)
    k = s[..., :1]
    tau = (off[-1] - off) / (p.substeps * YEAR)
    vol = implied_vol(p, var, tau)
    vol[..., 0] = ex["iv"]  # opened at the traded vol

    live = straddle(s[..., :-1], k, tau[:-1], vol[..., :-1])["value"]
    mark = np.concatenate([live, np.abs(s[..., -1:] - k)], axis=-1)
    iv = ex["iv"][..., None]
    delta = straddle(s[..., :-1], k, tau[:-1], iv)["delta"]
    moves = np.diff(mark, axis=-1) - delta * np.diff(s, axis=-1)
    return ex["units"][..., None] * moves


def by_month(p, daily):
    """(paths, days) to (paths, months, days in a straddle's life)."""
    months = p.years * YEAR // p.tenor
    return daily[:, : months * p.tenor].reshape(
        len(daily), months, p.tenor
    )


def realized_vol(p, log_s):
    """Annualized realized vol of each straddle-month, from closes."""
    returns = by_month(p, np.diff(log_s[:, :: p.substeps], axis=1))
    return np.sqrt((returns**2).sum(-1) * YEAR / p.tenor)


def mean_se(x):
    """Mean of x, with a standard error from the per-path means."""
    per_path = x.reshape(len(x), -1).mean(axis=1)
    se = per_path.std(ddof=1) / math.sqrt(len(per_path))
    return float(x.mean()), float(se)


FREQS = (0.2, 1.0, 2.0, 4.0, 8.0, 16.0)  # hedges per day


def premium_settings(p):
    """No premium, each part of the premium on its own, and both."""
    return {
        "fair": fair(p),
        "diffusive": replace(p, lam_q=p.lam_p),
        "jump": replace(p, theta_q=p.theta_p),
        "base": p,
    }


def run_chunk(p, chunk, light=False):
    """Every array from one chunk. light runs the base case only."""
    log_s, v, crash = simulate(p, chunk)
    months = by_month(p, crash)
    out = {"rv": realized_vol(p, log_s), "jump_m": months.any(-1)}
    if light:
        return out | {"base": monthly(p, explain(p, log_s, v))}

    out["jump_day"] = months.reshape(len(crash), -1)
    out["expiry_jump_m"] = months[..., -1]  # a crash on the last day
    out |= premium_runs(p, log_s, v)
    out |= frequency_runs(p, chunk, log_s, v)
    return out


def premium_runs(p, log_s, v):
    """The four premium settings, and the base case day by day."""
    out = {}
    for name, q in premium_settings(p).items():
        ex = explain(q, log_s, v)
        out[name] = monthly(q, ex)
        if name == "base":
            out |= daily_pnl(q, log_s, v, ex)
    return out


def daily_pnl(p, log_s, v, ex):
    """The long straddle's P&L per day, held and marked, and costs."""
    cost = ex["hedge_cost"].copy()
    cost[..., 0] -= p.option_cost_vol * p.vega_notional
    mtm = mark_to_market(p, log_s, v, ex)

    n = len(log_s)
    return {
        "day_held": ex["pnl"].reshape(n, -1),
        "day_mtm": mtm.reshape(n, -1),
        "day_cost": cost.reshape(n, -1),
    }


def frequency_runs(p, chunk, log_s, v):
    """Residual, costs and P&L at each frequency, jumps or none."""
    twin = simulate(no_jumps(p), chunk)[:2]
    worlds = {"jumps": (p, (log_s, v)), "no_jumps": (no_jumps(p), twin)}
    keep = ("residual", "hedge_cost", "pnl")

    out = {}
    for f in FREQS:
        for label, (q, paths) in worlds.items():
            m = monthly(
                q, explain(replace(q, hedges_per_day=f), *paths)
            )
            out[f"{label}_{f:g}"] = {x: m[x] for x in keep}
    return out


def join(parts):
    """Join equal nested dicts of arrays along the paths axis."""
    if isinstance(parts[0], dict):
        return {x: join([q[x] for q in parts]) for x in parts[0]}
    return np.concatenate(parts)


def run_all(p, light=False):
    """Every chunk of paths, joined."""
    return join([run_chunk(p, c, light) for c in range(n_chunks(p))])


def fair_short_gross(p):
    """The short straddle's mean P&L before costs with no premium."""
    q = fair(p)
    pnl = []
    for c in range(n_chunks(p)):
        log_s, v, _ = simulate(q, c)
        pnl.append(monthly(q, explain(q, log_s, v))["pnl"])
    return mean_se(-np.concatenate(pnl))


def fair_checks(p, d):
    """The no-premium mean on other paths and other pricing draws."""
    paths = [fair_short_gross(replace(p, seed=x)) for x in (1, 2, 3)]
    redraws = [replace(p, pricing_seed=x) for x in (7, 11, 2024)]
    pricing = [fair_short_gross(q) for q in redraws]

    pooled = [mean_se(-d["fair"]["pnl"])] + paths
    mean = sum(m for m, _ in pooled) / len(pooled)
    se = math.sqrt(sum(s**2 for _, s in pooled)) / len(pooled)
    return {
        "fair_check_paths_min": min(m for m, _ in paths),
        "fair_check_paths_max": max(m for m, _ in paths),
        "fair_check_pricing_min": min(m for m, _ in pricing),
        "fair_check_pricing_max": max(m for m, _ in pricing),
        "fair_pooled_mean": mean,
        "fair_pooled_t": mean / se,
    }


def pnl_stats(x):
    """Mean, standard error and the shape of a P&L distribution."""
    mean, se = mean_se(x)
    f = x.ravel()

    return {
        "mean": mean,
        "se": se,
        "median": float(np.median(f)),
        "std": float(f.std(ddof=1)),
        "skew": float(skew(f)),
        "win_pct": float(100 * np.mean(f > 0)),
        "p5": float(np.percentile(f, 5)),
        "worst": float(f.min()),
    }


def r2(y, x):
    """The share of y's variance explained by a fitted line in x."""
    return float(np.corrcoef(y.ravel(), x.ravel())[0, 1] ** 2)


def r2_face(y, x):
    """The share of y's variance removed by subtracting x as it is."""
    resid = y.ravel() - x.ravel()
    return float(1 - np.var(resid, ddof=1) / np.var(y.ravel(), ddof=1))


def explain_stats(p, d):
    """How well each explanation fits the long straddle's months."""
    b, jm = d["base"], d["jump_m"]
    pnl, gt, he = b["pnl"], b["gamma_theta"], b["residual"]
    naive = p.vega_notional * 100 * (d["rv"] - b["iv"])

    fits = {
        "r2_gamma_theta": r2(pnl, gt),
        "r2_gamma_theta_no_jump_months": r2(pnl[~jm], gt[~jm]),
        "r2_naive": r2(pnl, naive),
        "r2_face_gamma_theta": r2_face(pnl, gt),
        "r2_face_gamma_theta_no_jump_months": r2_face(
            pnl[~jm], gt[~jm]
        ),
        "r2_face_naive": r2_face(pnl, naive),
    }

    # the residual, and how much of it the jump months carry
    share = 100 * np.sum(he[jm] ** 2) / np.sum(he**2)
    return fits | {
        "pnl_std": float(pnl.std(ddof=1)),
        "hedge_error_std": float(he.std(ddof=1)),
        "hedge_error_var_share_jump_months_pct": float(share),
        "hedge_error_mean_jump_months": float(he[jm].mean()),
        "jump_month_pct": float(100 * jm.mean()),
        "expiry_jump_month_pct": float(100 * d["expiry_jump_m"].mean()),
    }


def frequency_stats(d):
    """One row per hedge frequency."""
    rows = []
    for f in FREQS:
        jumps, calm = d[f"jumps_{f:g}"], d[f"no_jumps_{f:g}"]
        ex_expiry = jumps["residual"][~d["expiry_jump_m"]]

        row = {"hedges_per_day": f}
        row["he_std_no_jumps"] = float(calm["residual"].std(ddof=1))
        row["he_std_jumps"] = float(jumps["residual"].std(ddof=1))
        row["he_std_jumps_ex_expiry"] = float(ex_expiry.std(ddof=1))
        row["pnl_std_jumps"] = float(jumps["pnl"].std(ddof=1))
        row["hedge_cost"] = float(jumps["hedge_cost"].mean())
        rows.append(row)
    return rows


def premium_stats(d):
    """Implied against realized vol, and both sides of the trade."""
    b, rv = d["base"], d["rv"]
    gap = 100 * (b["iv"] - rv)  # implied minus realized, vol points
    edges = range(-30, 16)
    clipped = np.clip(gap, -30 + 1e-9, 15 - 1e-9)

    vols = {
        "iv_mean": float(100 * b["iv"].mean()),
        "rv_mean": float(100 * rv.mean()),
        "gap_vol_pts": mean_se(gap),
        "gap_var_pts": mean_se(1e4 * (b["iv"] ** 2 - rv**2)),
        "rv_below_iv_pct": float(100 * np.mean(rv < b["iv"])),
    }

    return vols | {
        "short": pnl_stats(b["short"]),
        "long": pnl_stats(b["long"]),
        "hist_edges": list(edges),
        "hist_counts": np.histogram(clipped, edges)[0].tolist(),
    }


def crash_stats(d):
    """Losses on crash days, and the worst months and days."""
    short, jm, jd = d["base"]["short"], d["jump_m"], d["jump_day"]
    day_held, day_mtm = -d["day_held"], -d["day_mtm"]  # the short side
    gains = short[short > 0].sum()
    lost = np.minimum(short[jm], 0).sum()
    on_day = day_held[jd].sum()
    shares = {
        "lost_share_of_gains_pct": float(-100 * lost / gains),
        "jump_day_pct": float(100 * jd.mean()),
        "jump_day_share_of_gains_pct": float(-100 * on_day / gains),
    }

    has_jump = jd.any(axis=1)
    worst = short.min(axis=1)
    held = np.median(day_held.min(axis=1)[has_jump])
    mtm = np.median(day_mtm.min(axis=1)[has_jump])
    in_months = np.median(-worst / short.mean())
    worst_days = {
        "median_worst_month": float(np.median(worst)),
        "median_worst_month_in_avg_months": float(in_months),
        "median_worst_day_held_jump_paths": float(held),
        "median_worst_day_mtm_jump_paths": float(mtm),
        "paths_with_jump_pct": float(100 * has_jump.mean()),
    }

    return shares | worst_days | representative_path(d)


def representative_path(d):
    """The median path among those with a crash, day by day."""
    day_held, day_mtm = -d["day_held"], -d["day_mtm"]  # the short side
    cost = d["day_cost"]
    has_jump = d["jump_day"].any(axis=1)
    total = (day_held + cost).sum(axis=1)
    by_total = np.argsort(total[has_jump], kind="stable")
    order = np.flatnonzero(has_jump)[by_total]
    rep = int(order[len(order) // 2])

    held = np.cumsum(day_held[rep] + cost[rep])
    mtm = np.cumsum(day_mtm[rep] + cost[rep])
    drawdown = np.max(np.maximum.accumulate(held) - held)
    return {
        "rep_path": rep,
        "rep_jump_days": np.flatnonzero(d["jump_day"][rep]).tolist(),
        "rep_cum_held": np.round(held, 2).tolist(),
        "rep_cum_mtm": np.round(mtm, 2).tolist(),
        "rep_worst_day_held": float(day_held[rep].min()),
        "rep_max_drawdown_held": float(drawdown),
    }


def settings_stats(p, d):
    """The short straddle under each premium setting."""
    out = {}
    for name in premium_settings(p):
        out[name] = {
            "short": pnl_stats(d[name]["short"]),
            "short_pre_cost": pnl_stats(-d[name]["pnl"]),
            "iv_mean": float(100 * d[name]["iv"].mean()),
        }
    return out


def sensitivity(p, d):
    """Crashes twice as often in reality, with the same pricing rule."""
    d2 = run_all(replace(p, lam_p=2 * p.lam_p), light=True)
    short = d2["base"]["short"]
    lost = np.minimum(short[d2["jump_m"]], 0).sum()
    gains = short[short > 0].sum()

    stats = {
        "short": pnl_stats(short),
        "lost_share_of_gains_pct": float(-100 * lost / gains),
    }
    return stats, short - d["base"]["short"]


def model_numbers(p):
    """The settings as the text quotes them."""
    t = p.tenor / YEAR
    v_long = long_run_var(p, p.theta_p, p.lam_p)

    world = {
        "theta_p_vol": 100 * math.sqrt(p.theta_p),
        "theta_q_vol": 100 * math.sqrt(p.theta_q),
        "half_life_months": 12 * math.log(2) / p.kappa,
        "rho_abs": abs(p.rho),
        "dv": p.dv,
        "crash_every_years_p": 1 / p.lam_p,
        "crash_every_years_q": 1 / p.lam_q,
        "mu_j_pct": 100 * p.mu_j,
        "sig_j_pct": 100 * p.sig_j,
    }

    return world | {
        "paths": p.paths,
        "straddle_months": p.paths * (p.years * YEAR // p.tenor),
        "vega_notional": p.vega_notional,
        "lr_varswap_vol": float(100 * varswap_vol(p, v_long, t)),
        "lr_atm_vol": float(100 * implied_vol(p, v_long, t)),
        "option_cost_per_month": p.option_cost_vol * p.vega_notional,
    }


def explain_numbers(e, freq):
    """Ratios for the first two findings."""
    f1, f16 = freq[1.0], freq[16.0]
    calm = f16["he_std_no_jumps"] / f1["he_std_no_jumps"]
    ex_expiry = (
        f16["he_std_jumps_ex_expiry"] / f1["he_std_jumps_ex_expiry"]
    )
    pnl_std = f16["pnl_std_jumps"] / f1["pnl_std_jumps"]
    cost = f16["hedge_cost"] / f1["hedge_cost"]

    fits = {
        "r2_face_gamma_theta_pct": 100 * e["r2_face_gamma_theta"],
        "he_to_pnl_std_pct": 100 * e["hedge_error_std"] / e["pnl_std"],
    }

    return fits | {
        "freq16_to_1_no_jumps_pct": 100 * calm,
        "he_jumps_min": min(r["he_std_jumps"] for r in freq.values()),
        "freq1_he_jumps_ex_expiry": f1["he_std_jumps_ex_expiry"],
        "freq16_he_jumps_ex_expiry": f16["he_std_jumps_ex_expiry"],
        "freq16_vs_1_jumps_ex_expiry_rise_pct": 100 * ex_expiry - 100,
        "freq16_to_1_pnl_std_pct": 100 * pnl_std,
        "freq16_to_1_cost_multiple": cost,
        "freq1_cost": f1["hedge_cost"],
        "freq16_cost": f16["hedge_cost"],
    }


def premium_numbers(pr, s):
    """Numbers for the premium finding."""
    iv = {k: x["iv_mean"] for k, x in s.items()}
    jump_part = iv["jump"] ** 2 - iv["fair"] ** 2
    whole = iv["base"] ** 2 - iv["fair"] ** 2
    short = pr["short"]
    to_median = 100 * short["mean"] / short["median"]

    gaps = {
        "base_iv_mean": pr["iv_mean"],
        "rv_mean": pr["rv_mean"],
        "gap_vol_pts": pr["gap_vol_pts"][0],
        "gap_vol_pts_se": pr["gap_vol_pts"][1],
        "gap_var_pts": pr["gap_var_pts"][0],
        "jump_share_of_premium_pct": 100 * jump_part / whole,
    }

    return gaps | {
        "base_short_gross_mean": s["base"]["short_pre_cost"]["mean"],
        "short_net_mean": short["mean"],
        "short_net_se": short["se"],
        "short_skew_abs": abs(short["skew"]),
        "short_mean_over_median_pct": to_median,
    }


def crash_numbers(p, c, sens, change):
    """Numbers for the crash finding and its sensitivity."""
    held = c["median_worst_day_held_jump_paths"]
    mtm = c["median_worst_day_mtm_jump_paths"]
    path = c["rep_cum_held"]
    carry = c["median_worst_month_in_avg_months"]

    days = {
        "jump_day_pct": c["jump_day_pct"],
        "lost_share_of_gains_pct": c["lost_share_of_gains_pct"],
        "jump_day_share_of_gains_pct": c["jump_day_share_of_gains_pct"],
        "months_of_carry_in_worst_month": carry,
        "rep_end_below_peak": path[-1] - max(path),
        "worst_day_mtm_to_held": mtm / held,
    }

    return days | {
        "sens_lam_crash_every_years": 1 / (2 * p.lam_p),
        "sens_short_mean": sens["short"]["mean"],
        "sens_short_se": sens["short"]["se"],
        "sens_short_win_pct": sens["short"]["win_pct"],
        "sens_change_short_mean": change[0],
        "sens_change_short_mean_se": change[1],
        "sens_lost_share_of_gains_pct": sens["lost_share_of_gains_pct"],
    }


def fair_numbers(s):
    """The no-premium short straddle, before and after costs."""
    gross = s["fair"]["short_pre_cost"]
    base_gross = s["base"]["short_pre_cost"]["mean"]
    share = 100 * gross["mean"] / base_gross

    return {
        "fair_short_win_pct": s["fair"]["short"]["win_pct"],
        "fair_short_gross_mean": gross["mean"],
        "fair_short_gross_se": gross["se"],
        "fair_gross_share_of_base_pct": share,
        "fair_gross_t": gross["mean"] / gross["se"],
        "fair_short_net_mean": s["fair"]["short"]["mean"],
    }


def change_numbers(d, s):
    """No premium against the full premium, on the same paths."""
    fair_short, base_short = d["fair"]["short"], d["base"]["short"]
    wins = (base_short > 0).astype(float) - (fair_short > 0)
    win_change = mean_se(100 * wins)
    change = mean_se(base_short - fair_short)
    base_win = s["base"]["short"]["win_pct"]
    fair_win = s["fair"]["short"]["win_pct"]

    return {
        "base_short_win_pct": base_win,
        "win_pct_change_fair_to_base": base_win - fair_win,
        "win_pct_change_fair_to_base_se": win_change[1],
        "short_net_change_fair_to_base": change[0],
        "short_net_change_fair_to_base_se": change[1],
    }


def money(x):
    """Whole dollars with a minus sign for losses."""
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def signed(x, fmt):
    """A number with a minus sign rather than a hyphen."""
    return format(x, fmt).replace("-", "−")


def table(rows, widths):
    """Right-aligned columns of the given widths."""
    lines = []
    for row in rows:
        cells = zip(row, widths, strict=True)
        lines.append("  ".join(str(c).rjust(w) for c, w in cells))
    return "\n".join(lines)


EXPLANATIONS = (
    ("gamma-theta P&L (summed daily)", "gamma_theta"),
    (
        "the same, in months without a crash",
        "gamma_theta_no_jump_months",
    ),
    ("vega × (realized vol − implied vol)", "naive"),
)


def explain_table(e):
    """R² and the share at face value of each explanation."""
    label = "explanation of the month's hedged P&L"
    rows = [[label, "R² (squared", "share at"]]
    rows.append(["", "correlation)", "face value"])
    for label, key in EXPLANATIONS:
        fit, face = e[f"r2_{key}"], e[f"r2_face_{key}"]
        rows.append([label, signed(fit, ".2f"), signed(face, ".2f")])
    return table(rows, [40, 12, 12])


def frequency_table(freq):
    """The residual, P&L and costs at each hedge frequency."""
    std = "residual std"
    rows = [["hedges", std, std, "P&L std", "hedging"]]
    rows.append(["per day", "no jumps", "with jumps", "with jumps"])
    rows[-1].append("cost/month")
    keys = ("he_std_no_jumps", "he_std_jumps", "pnl_std_jumps")
    for f, r in freq.items():
        cells = [money(r[k]) for k in keys + ("hedge_cost",)]
        rows.append([f"{f:g}"] + cells)
    return table(rows, [7, 13, 13, 12, 11])


SIDES = (
    ("mean", "mean", money),
    ("standard error", "se", money),
    ("median", "median", money),
    ("standard deviation", "std", money),
    ("skewness", "skew", lambda x: signed(x, ".1f")),
    ("months that made money", "win_pct", lambda x: f"{x:.0f}%"),
    ("5th percentile month", "p5", money),
    ("worst month", "worst", money),
)


def short_long_table(pr):
    """Both sides of the trade per straddle-month, after costs."""
    rows = [["per straddle-month", "short", "long"]]
    for label, key, fmt in SIDES:
        rows.append(
            [label, fmt(pr["short"][key]), fmt(pr["long"][key])]
        )
    return table(rows, [26, 13, 13])


SETTINGS = (
    ("fair", "none (Q = P)"),
    ("diffusive", "diffusive only"),
    ("jump", "jump only"),
    ("base", "both (base case)"),
)


def settings_table(s):
    """The short straddle under each premium setting."""
    head = ["premium in the price", "win rate", "mean, gross"]
    rows = [head + ["mean, net ± SE", "5th pct, net"]]
    for k, label in SETTINGS:
        net, gross = s[k]["short"], s[k]["short_pre_cost"]
        spread = f"{money(net['mean'])} ± {net['se']:,.0f}"
        win = f"{net['win_pct']:.0f}%"
        mean = money(gross["mean"])
        rows.append([label, win, mean, spread, money(net["p5"])])
    return table(rows, [22, 9, 12, 15, 13])


def all_tables(res, freq):
    """The four text tables of the article."""
    return {
        "explain_r2": explain_table(res["explain"]),
        "frequency": frequency_table(freq),
        "short_long": short_long_table(res["premium"]),
        "settings": settings_table(res["settings"]),
    }


def scatter_data(d):
    """Each month's P&L, its gamma-theta P&L and its crash flag."""
    b = d["base"]
    return {
        "x": b["gamma_theta"].ravel(),
        "y": b["pnl"].ravel(),
        "jump_month": d["jump_m"].ravel(),
    }


def results(p=BASE):
    """Every number and table in the article, and the scatter data."""
    d = run_all(p)
    res = {
        "config": asdict(p),
        "explain": explain_stats(p, d),
        "frequency": frequency_stats(d),
        "premium": premium_stats(d),
        "crash": crash_stats(d),
        "settings": settings_stats(p, d),
    }

    res["sensitivity"], change = sensitivity(p, d)
    freq = {r["hedges_per_day"]: r for r in res["frequency"]}
    scatter = scatter_data(d)

    e, pr, s = res["explain"], res["premium"], res["settings"]
    sens = mean_se(change)

    res["derived"] = (
        model_numbers(p)
        | explain_numbers(e, freq)
        | premium_numbers(pr, s)
        | crash_numbers(p, res["crash"], res["sensitivity"], sens)
        | fair_numbers(s)
        | fair_checks(p, d)
        | change_numbers(d, s)
    )
    res["tables"] = all_tables(res, freq)
    return res, scatter


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

GREY, GRID, SURFACE = "#52514e", "#e6e6e6", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
rc = plt.rcParams

# every chart is 7 inches at 200 dpi, which is 1400 pixels wide
rc["figure.figsize"] = (7, 3.9)
rc["savefig.dpi"] = 200
rc["figure.constrained_layout.use"] = True
rc["figure.constrained_layout.h_pad"] = 0.12
rc["figure.facecolor"] = rc["axes.facecolor"] = SURFACE

# one font in two sizes, with titles in bold on the left
rc["font.family"] = "sans-serif"
rc["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
rc["font.size"] = 10
rc["axes.titlesize"] = 12
rc["axes.titleweight"] = "bold"
rc["axes.titlelocation"] = "left"


# no axis lines and no tick marks, only light horizontal grid lines
for side in ("top", "right", "left", "bottom"):
    rc[f"axes.spines.{side}"] = False
rc["axes.grid"] = True
rc["axes.grid.axis"] = "y"
rc["axes.axisbelow"] = True
rc["grid.color"] = GRID
rc["grid.linewidth"] = 0.8
rc["xtick.major.size"] = rc["ytick.major.size"] = 0
rc["xtick.major.pad"] = 6

# quiet grey text around the data, and lines 1.5 points wide
rc["xtick.color"] = rc["ytick.color"] = GREY
rc["axes.labelcolor"] = GREY
rc["lines.linewidth"] = 1.5
rc["lines.markersize"] = 5
rc["legend.frameon"] = False


def dollars(axis, unit=None):
    """Label an axis in dollars, every tick in the same form."""
    ticks = [t for t in axis.get_majorticklocs() if t != 0]
    top = max((abs(t) for t in ticks), default=1)
    if unit is None:
        unit = "M" if top >= 1e6 else "k" if top >= 1e4 else ""
    scale = {"M": 1e6, "k": 1e3, "": 1}[unit]
    shown = [f"{abs(t) / scale:g}" for t in ticks]
    digits = max((len(s.partition(".")[2]) for s in shown), default=0)

    def label(x, _):
        if x == 0:
            return "$0"
        sign = "−" if x < 0 else ""
        return f"{sign}${abs(x) / scale:,.{digits}f}{unit}"

    axis.set_major_formatter(FuncFormatter(label))


def finish(ax, title, money=False, legend=False):
    """Title, round y ticks, a zero line and the legend of a panel."""
    ax.set_title(title, pad=30 if legend else 12)

    lo, hi = ax.dataLim.intervaly
    steps = MaxNLocator(6, steps=[1, 2, 5, 10])
    ticks = steps.tick_values(lo, hi)
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0], ticks[-1])
    if ticks[0] <= 0 <= ticks[-1]:
        ax.axhline(0, color=GREY, lw=0.8, zorder=1)
    if money:
        dollars(ax.yaxis)

    if legend:
        n = len(ax.get_legend_handles_labels()[0])
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1),
            ncols=n,
            borderaxespad=0.3,
            handlelength=1.4,
            columnspacing=1.4,
        )


def save(fig, path):
    """Write a chart without software metadata and close it."""
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def scatter_axes(ax, x, title, legend=False):
    """Round ticks in one dollar unit, and the line where y equals x."""
    finish(ax, title, money=True, legend=legend)
    steps = MaxNLocator(4, steps=[1, 2, 5, 10])
    ticks = steps.tick_values(x.min(), x.max())
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_xlabel("gamma-theta P&L, summed daily")

    # both axes in the same unit, $k or $M
    y_lo, y_hi = ax.get_ylim()
    top = max(abs(ticks[0]), ticks[-1], abs(y_lo), y_hi)
    unit = "M" if top >= 1e6 else "k"
    dollars(ax.xaxis, unit)
    dollars(ax.yaxis, unit)

    # where the P&L equals the gamma-theta P&L, inside both axes
    lo, hi = max(ticks[0], y_lo), min(ticks[-1], y_hi)
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8)


def scatter_chart(sc, path):
    """Each month's hedged P&L against its gamma-theta P&L."""
    x, y, jm = sc["x"], sc["y"], sc["jump_month"]
    fig, (wide, near) = plt.subplots(1, 2)

    # calm months are drawn on top of the crash months
    calm = dict(s=5, color=BLUE, alpha=0.25, lw=0, rasterized=True)
    crash = dict(s=8, color=ORANGE, alpha=0.7, lw=0, rasterized=True)
    wide.scatter(x[~jm], y[~jm], label="no crash", zorder=3, **calm)
    wide.scatter(x[jm], y[jm], label="crash", **crash)
    near.scatter(x[~jm], y[~jm], **calm)

    scatter_axes(wide, x, "Hedged P&L, all months", legend=True)
    scatter_axes(near, x[~jm], "Hedged P&L, calm months")
    near.set_title(near.get_title("left"), pad=30)  # level with left

    for handle in wide.get_legend().legend_handles:
        handle.set_sizes([30])
        handle.set_alpha(1)
    save(fig, path)


def frequency_chart(res, path):
    """The residual's standard deviation at each hedge frequency."""
    rows = res["frequency"]
    at = np.arange(len(rows))
    fig, ax = plt.subplots()

    for key, colour, label in (
        ("he_std_no_jumps", BLUE, "no jumps"),
        ("he_std_jumps", ORANGE, "with jumps"),
        (
            "he_std_jumps_ex_expiry",
            GREEN,
            "with jumps, no crash on the last day",
        ),
    ):
        values = [r[key] for r in rows]
        ax.plot(at, values, color=colour, marker="o", label=label)

    ax.set_xticks(at, [f"{r['hedges_per_day']:g}" for r in rows])
    ax.set_xlim(-0.3, len(rows) - 0.7)
    ax.set_xlabel("hedges per day")
    title = "Standard deviation of the monthly residual"
    finish(ax, title, money=True, legend=True)
    save(fig, path)


def premium_chart(res, path):
    """Straddle-months by implied minus realized vol."""
    pr = res["premium"]
    edges = np.array(pr["hist_edges"])
    counts = np.array(pr["hist_counts"])
    centers = edges[:-1] + 0.5
    above = centers > 0

    fig, ax = plt.subplots()
    bars = dict(width=1.0, edgecolor=SURFACE, lw=0.8)
    up, down = "implied above realized", "realized above implied"
    ax.bar(centers[above], counts[above], color=BLUE, label=up, **bars)
    ax.bar(
        centers[~above],
        counts[~above],
        color=ORANGE,
        label=down,
        **bars,
    )

    ax.set_xticks(range(-30, 16, 5))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("implied vol minus realized vol, in vol points")
    title = "Straddle-months by implied minus realized vol"
    finish(ax, title, legend=True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    save(fig, path)


def crash_chart(res, path):
    """Cumulative P&L of the short straddle on one path."""
    c = res["crash"]
    held = np.array(c["rep_cum_held"])
    mtm = np.array(c["rep_cum_mtm"])
    years = (np.arange(len(held)) + 1) / YEAR

    fig, ax = plt.subplots()
    ax.plot(years, held, color=BLUE, label="held to expiry", zorder=3)
    ax.plot(
        years, mtm, color=ORANGE, label="marked at current implied vol"
    )
    for i, day in enumerate(c["rep_jump_days"]):
        label = "crash day" if i == 0 else None
        x = (day + 1) / YEAR
        ax.axvline(x, color=GREY, lw=0.8, ls=(0, (3, 3)), label=label)

    ax.set_xticks(range(6))
    ax.set_xlim(0, 5)
    ax.set_xlabel("years")
    title = "Cumulative P&L of the short straddle after costs"
    finish(ax, title, money=True, legend=True)
    save(fig, path)


def main(out=Path(".")):
    res, scatter = results()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scatter_chart(scatter, figures / "explain_scatter.png")
    frequency_chart(res, figures / "hedge_frequency.png")
    premium_chart(res, figures / "premium_hist.png")
    crash_chart(res, figures / "crash_path.png")

    text = json.dumps(res, indent=1, default=float)
    (out / "results.json").write_text(text + "\n")
    print(*res["tables"].values(), sep="\n\n")


if __name__ == "__main__":
    main()
