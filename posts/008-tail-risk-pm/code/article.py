# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr
from scipy.stats import poisson

MONTH = 1 / 12  # one step of the simulation, in years
FLOOR = 1e-9  # a gross return this low would wipe a portfolio out


def crash_size(p):
    """Average fall of the index in one crash, as a fraction."""
    return math.exp(p.jump_mean + p.jump_sd**2 / 2) - 1


def month_returns(p, draws, crash_rate):
    """Gross index returns S(t+1) / S(t), one per path and month.

    The three draws are a normal shock for the calm moves, a uniform
    that sets the number of crashes and a normal shock for their size.
    """
    calm, uniform, size = draws

    # crashes in each month, by inverting the Poisson distribution,
    # so no month's count falls when the crash rate rises
    cdf = poisson.cdf(np.arange(30), crash_rate * MONTH)
    crashes = np.searchsorted(cdf, uniform, side="right")

    # the calm drift pays for the crashes, so the mean stays e^(mu/12)
    drift = p.mu - crash_rate * crash_size(p) - p.sigma**2 / 2
    log_x = drift * MONTH + p.sigma * math.sqrt(MONTH) * calm
    jumps = crashes * p.jump_mean + p.jump_sd * np.sqrt(crashes) * size
    return np.exp(log_x + jumps), crashes


def bs_put(s, k, tau, sigma, r):
    """Black-Scholes price of a European put."""
    root = sigma * np.sqrt(tau)
    d1 = (np.log(s / k) + (r + sigma**2 / 2) * tau) / root
    d2 = d1 - root
    return k * np.exp(-r * tau) * ndtr(-d2) - s * ndtr(-d1)


def merton_put(p, strike, crash_rate=None, tau=MONTH):
    """Merton's price of a put on an index at 1 when the market
    prices `crash_rate` crashes a year.

    The price is a Poisson-weighted sum of Black-Scholes prices, one
    for each number of crashes before expiry.
    """
    lam = p.priced_rate if crash_rate is None else crash_rate
    k_bar = crash_size(p)
    n = np.arange(25)

    weights = poisson.pmf(n, lam * (1 + k_bar) * tau)
    sigma_n = np.sqrt(p.sigma**2 + n * p.jump_sd**2 / tau)
    r_n = p.r - lam * k_bar + n * math.log(1 + k_bar) / tau
    return float(weights @ bs_put(1.0, strike, tau, sigma_n, r_n))


def portfolio_return(p, x, exposure, hedge, strike=None):
    """Gross return of the portfolio over one month.

    It holds `exposure` times its value in the index, buys puts on
    `hedge` times that holding and keeps the rest in cash, borrowing
    when the cash is negative. At month end it rebalances.
    """
    k = p.strike if strike is None else strike
    premium = merton_put(p, k)
    cash = 1 - exposure - hedge * exposure * premium
    payoff = hedge * exposure * np.maximum(k - x, 0.0)
    return exposure * x + payoff + cash * math.exp(p.r * MONTH)


def expectation(p, f, crash_rate=None):
    """The exact mean of f(x) over one month's index return x.

    Given the number of crashes, the log return is normal, so the
    mean is a Poisson-weighted sum of integrals on a fine grid.
    """
    lam = p.crash_rate if crash_rate is None else crash_rate
    drift = p.mu - lam * crash_size(p) - p.sigma**2 / 2
    z = np.linspace(-10, 10, 8001)
    density = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)

    total = 0.0
    for n in range(9):
        mean = drift * MONTH + n * p.jump_mean
        sd = math.sqrt(p.sigma**2 * MONTH + n * p.jump_sd**2)
        values = f(np.exp(mean + sd * z)) * density
        weight = poisson.pmf(n, lam * MONTH)
        total = total + weight * np.trapezoid(values, z, axis=-1)
    return total


def growth_parts(p, exposure, hedge, crash_rate=None):
    """Average return, variance drag, tail term and growth rate, all
    in fractions a year. Growth = average - drag + tail, exactly."""

    def moments(x):
        r = portfolio_return(p, x, exposure, hedge)
        r = np.maximum(r, FLOOR)
        return np.array([r, r**2, np.log(r)])

    mean, square, log_mean = expectation(p, moments, crash_rate)
    average = mean - 1
    variance = square - mean**2
    return {
        "average": 12 * average,
        "drag": 12 * variance / 2,
        "tail": 12 * (log_mean - average + variance / 2),
        "growth": 12 * log_mean,
        "vol": math.sqrt(12 * variance),
    }


def drawdowns(log_wealth):
    """Deepest fall from a previous peak, and the longest stretch in
    months below a previous peak, for each path, on month-end values.
    """
    start = np.zeros((len(log_wealth), 1))
    path = np.hstack([start, log_wealth])
    peak = np.maximum.accumulate(path, axis=1)
    deepest = 1 - np.exp(np.min(path - peak, axis=1))

    # count the months since the last peak, and keep the longest run
    run = longest = np.zeros(len(path))
    for below in (path < peak).T:
        run = (run + 1) * below
        longest = np.maximum(longest, run)
    return deepest, longest


def growth_gap(p, exposure, crash_rate, strike=None):
    """Hedged minus unhedged growth a year, at one real crash rate."""
    q = p if strike is None else replace(p, strike=strike)
    hedged = growth_parts(q, exposure, q.hedge, crash_rate)
    plain = growth_parts(q, exposure, 0.0, crash_rate)
    return hedged["growth"] - plain["growth"]


def hedge_profit(p, crash_rate, strike=None):
    """The put's expected payoff minus its cost, per unit of index."""
    k = p.strike if strike is None else strike
    payoff = expectation(p, lambda x: np.maximum(k - x, 0), crash_rate)
    return payoff - merton_put(p, k) * math.exp(p.r * MONTH)


def break_even(f, top=1.0):
    """The crash rate between 0 and `top` where f crosses zero."""
    if f(0.0) * f(top) > 0:
        return math.nan
    return brentq(f, 0.0, top, xtol=1e-6)


@dataclass(frozen=True)
class Params:
    # the market, in yearly units with continuous compounding
    r: float = 0.03  # the cash rate
    mu: float = 0.08  # the index's expected total return
    sigma: float = 0.15  # volatility of the calm moves
    crash_rate: float = 0.10  # real crashes a year
    jump_mean: float = -0.20  # mean log size of a crash
    jump_sd: float = 0.07  # standard deviation of the log size

    # the options market prices crashes as twice as frequent
    priced_rate: float = 0.20

    # the protection
    strike: float = 0.90  # one-month puts, as a share of the index
    hedge: float = 1.0  # puts per unit of index held

    # the experiment
    years: int = 30
    paths: int = 20_000
    seed: int = 7
    drawdown_limit: float = 0.40  # on the 95th percentile
    high_rate: float = 0.25  # a crash rate above the priced one


BASE = Params()
STRIKES = (0.80, 0.85, 0.90, 0.95, 1.00)
EXPOSURES = np.round(np.arange(0, 2.01, 0.1), 1)
HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30)


def draw(p):
    """The random inputs of every path and month."""
    seeds = np.random.SeedSequence(p.seed).spawn(3)
    calm, uniform, size = (np.random.default_rng(s) for s in seeds)
    shape = (p.paths, 12 * p.years)
    return (
        calm.standard_normal(shape),
        uniform.random(shape),
        size.standard_normal(shape),
    )


def run(p, x, exposure, hedge):
    """One portfolio on every path, kept as per-path summaries."""
    r = portfolio_return(p, x, exposure, hedge)
    floored = int(np.sum(r < FLOOR))
    log_wealth = np.cumsum(np.log(np.maximum(r, FLOOR)), axis=1)
    deepest, underwater = drawdowns(log_wealth)

    # the puts' own profit, as a share of the portfolio, by year
    premium = merton_put(p, p.strike) * math.exp(p.r * MONTH)
    put = hedge * exposure * (np.maximum(p.strike - x, 0) - premium)
    years = put.reshape(len(x), p.years, 12).sum(axis=2)

    return {
        "final": log_wealth[:, -1],
        "yearly": log_wealth[:, 11::12],
        "average": 12 * (r.mean(axis=1) - 1),
        "worst": float(r.min() - 1),
        "deepest": deepest,
        "underwater": underwater,
        "put_years": years,
        "floored": floored,
    }


def mean_se(values):
    """Mean and standard error over independent paths."""
    v = np.asarray(values)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def paired(a, b, key="final", scale=None):
    """Mean and standard error of b minus a, path by path. Final log
    wealth is turned into growth a year."""
    if scale is None:
        scale = 1 / BASE.years if key == "final" else 1.0
    return mean_se(scale * (b[key] - a[key]))


def longest_losing_run(put_years):
    """The longest run of years in a row in which the puts lost."""
    run = longest = np.zeros(len(put_years))
    for lost in (put_years < 0).T:
        run = (run + 1) * lost
        longest = np.maximum(longest, run)
    return longest


def strike_row(p, k):
    """Premium, implied vol, payoff and break-evens of one strike."""
    price = merton_put(p, k)

    def gap(v):
        return bs_put(1.0, k, MONTH, v, p.r) - price

    row = {"strike": k, "premium": price, "vol": brentq(gap, 1e-3, 3)}
    row["payoff"] = hedge_profit(p, p.crash_rate, k) + price

    # the crash rates at which the puts pay, on average and in growth
    row["arith"] = break_even(lambda lam: hedge_profit(p, lam, k))
    row["geo"] = break_even(lambda lam: growth_gap(p, 1.0, lam, k))
    return row


def pricing(p):
    """One row for each strike in the strike table."""
    return [strike_row(p, k) for k in STRIKES]


def zero_check(p, draws):
    """Under the pricing measure the discounted mean payoff of the
    put must equal its price, so the gap must sit near zero."""
    q = replace(p, mu=p.r)
    x, _ = month_returns(q, draws, p.priced_rate)
    discount = math.exp(-p.r * MONTH)
    payoff = discount * np.maximum(p.strike - x.ravel(), 0)
    mean, se = mean_se(payoff)
    return mean - merton_put(p, p.strike), se


def experiments(p=BASE):
    """Every simulation the article reports, on one set of draws."""
    draws = draw(p)
    x, crashes = month_returns(p, draws, p.crash_rate)
    out = {"x": x, "crashes": crashes, "draws": draws}
    out["plain"] = run(p, x, 1.0, 0.0)
    out["hedged"] = run(p, x, 1.0, p.hedge)

    # an unhedged mix with the same average return as the hedged one
    a0, a1 = (growth_parts(p, w, 0.0)["average"] for w in (0, 1))
    hedged_average = growth_parts(p, 1.0, p.hedge)["average"]
    out["mix_exposure"] = (hedged_average - a0) / (a1 - a0)
    out["mix"] = run(p, x, out["mix_exposure"], 0.0)

    # every exposure, with and without the puts
    out["grid"] = {}
    for h in (0.0, p.hedge):
        for w in EXPOSURES:
            out["grid"][h, w] = run(p, x, w, h)
    out["zero"] = zero_check(p, draws)
    return out


def at_rate(p, draws, crash_rate):
    """Unhedged and hedged portfolios when crashes come at a
    different real rate, on the same draws."""
    x, _ = month_returns(p, draws, crash_rate)
    return run(p, x, 1.0, 0.0), run(p, x, 1.0, p.hedge)


def sweep(p):
    """Hedged minus unhedged growth on a grid of real crash rates."""
    rates = np.round(np.arange(0, 0.401, 0.01), 2)
    gaps = {}
    for w in (0.6, 1, 1.5):
        gaps[w] = [growth_gap(p, w, lam) for lam in rates]
    return rates, gaps


def market_numbers(p, r, prices):
    """Numbers for the sections on the market and the puts."""
    base = prices[STRIKES.index(p.strike)]
    out = {"r": 100 * p.r, "mu": 100 * p.mu, "sigma": 100 * p.sigma}
    out["crash_rate"] = p.crash_rate
    out["crash_years"] = 1 / p.crash_rate
    out["priced_rate"] = p.priced_rate
    out["crash_size"] = -100 * crash_size(p)
    out["paths"], out["years"] = p.paths, p.years
    out["strike"] = 100 * p.strike

    out["premium_month"] = 100 * base["premium"]
    out["premium_year"] = 1200 * base["premium"]
    out["payoff_month"] = 100 * base["payoff"]
    out["price_to_payoff"] = base["premium"] / base["payoff"]
    out["vol_90"] = 100 * base["vol"]
    out["vol_100"] = 100 * prices[-1]["vol"]
    out["vol_80"] = 100 * prices[0]["vol"]

    pays = float(np.mean(r["x"] < p.strike))
    out["put_pays_pct"], out["put_pays_every"] = 100 * pays, 1 / pays
    out["zero_z"] = abs(r["zero"][0]) / r["zero"][1]
    return out


def compound_numbers(parts, r):
    """Average and compounded returns, unhedged and hedged."""
    out = {}
    for key in ("average", "drag", "tail", "growth", "vol"):
        for name in ("plain", "hedged"):
            out[f"{name}_{key}"] = 100 * parts[name][key]
        diff = parts["hedged"][key] - parts["plain"][key]
        out[f"diff_{key}"] = 100 * diff
    out["drag_saved"] = -out["diff_drag"]

    # what the puts cost in basis points a year, path by path
    for name, key in (("arith", "average"), ("geo", "final")):
        cost, se = paired(r["hedged"], r["plain"], key)
        out[f"{name}_cost"] = 1e4 * cost
        out[f"{name}_cost_se"] = 1e4 * se
    share = out["geo_cost"] / out["arith_cost"]
    out["geo_share_of_arith"] = 100 * share
    return out


def path_numbers(p, r, parts):
    """Simulated growth, worst months, wealth and the puts' years."""
    out = {}
    for name in ("plain", "hedged"):
        final = r[name]["final"]
        g, se = mean_se(final / p.years)
        out[f"{name}_quad_z"] = abs(g - parts[name]["growth"]) / se
        out[f"{name}_worst"] = -100 * r[name]["worst"]
        wealth = np.exp(np.percentile(final, [5, 50]))
        out[f"{name}_p5_wealth"] = float(wealth[0])
        out[f"{name}_median_wealth"] = float(wealth[1])

    # the puts' own profit in each year, as a share of the portfolio
    years = r["hedged"]["put_years"]
    out["put_years_lost_pct"] = 100 * float(np.mean(years < 0))
    longest = longest_losing_run(years)
    out["losing_run_median"] = float(np.median(longest))
    return out


def drawdown_numbers(r):
    """Drawdowns of the unhedged, hedged and static-mix portfolios."""
    out = {"mix_exposure": 100 * r["mix_exposure"]}
    for name in ("plain", "hedged", "mix"):
        deep = 100 * r[name]["deepest"]
        out[f"{name}_dd_median"] = float(np.median(deep))
        out[f"{name}_dd_p95"] = float(np.percentile(deep, 95))
        out[f"{name}_dd_30"] = 100 * float(np.mean(deep > 30))
        out[f"{name}_dd_50"] = 100 * float(np.mean(deep > 50))
        under = np.median(r[name]["underwater"]) / 12
        out[f"{name}_under_years"] = float(under)

    # the hedged portfolio against the mix, path by path
    dd, dd_se = paired(r["mix"], r["hedged"], "deepest", 100)
    out["dd_vs_mix"], out["dd_vs_mix_se"] = dd, dd_se
    growth, growth_se = paired(r["hedged"], r["mix"])
    out["mix_vs_hedged_growth"] = 1e4 * growth
    out["mix_vs_hedged_growth_se"] = 1e4 * growth_se
    return out


def break_even_numbers(p, prices, gaps, check):
    """Break-even crash rates, and one growth gap checked by
    simulation."""
    base = prices[STRIKES.index(p.strike)]
    out = {"arith_be": base["arith"], "geo_be": base["geo"]}
    out["arith_be_years"] = 1 / base["arith"]
    out["geo_be_years"] = 1 / base["geo"]
    out["atm_be_years"] = 1 / prices[-1]["geo"]
    for w in (0.6, 1.5):
        be = break_even(lambda lam: growth_gap(p, w, lam))
        out[f"geo_be_{round(10 * w)}"] = be

    # growth the puts cost, unhedged minus hedged
    out["cost_no_crash"] = -100 * gaps[1][0]
    lam, plain, hedged = check
    out["check_rate"] = lam
    out["check_quad"] = -1e4 * growth_gap(p, 1.0, lam)
    cost, se = paired(hedged, plain)
    out["check_sim"], out["check_se"] = 1e4 * cost, 1e4 * se
    return out


def kelly(p, hedge):
    """The growth-optimal exposure and its growth a year."""
    best = minimize_scalar(
        lambda w: -growth_parts(p, w, hedge)["growth"],
        bounds=(0, 4),
        method="bounded",
    )
    return float(best.x), -float(best.fun)


def largest_within_limit(p, r, hedge):
    """The largest exposure on the grid whose 95th-percentile
    30-year drawdown stays within the limit."""
    ok = []
    for w in EXPOSURES:
        deep = r["grid"][hedge, w]["deepest"]
        if np.percentile(deep, 95) <= p.drawdown_limit:
            ok.append(float(w))
    return max(ok)


def exposure_numbers(p, r):
    """Kelly and drawdown-limited exposures, unhedged and hedged."""
    out, limited = {}, {}
    for name, h in (("plain", 0.0), ("hedged", p.hedge)):
        w, g = kelly(p, h)
        out[f"{name}_kelly"], out[f"{name}_kelly_growth"] = w, 100 * g
        w = largest_within_limit(p, r, h)
        limited[name] = r["grid"][h, w]
        out[f"{name}_limit_w"] = w
        out[f"{name}_worst_2"] = -100 * r["grid"][h, 2.0]["worst"]

    out["limit"] = 100 * p.drawdown_limit
    cost, se = paired(limited["hedged"], limited["plain"])
    out["limit_cost"], out["limit_cost_se"] = 1e4 * cost, 1e4 * se
    out["floored_all"] = sum(x["floored"] for x in r["grid"].values())
    return out


def ahead(plain, hedged):
    """Percent of paths where hedged wealth is ahead, per horizon."""
    lead = hedged["yearly"] > plain["yearly"]
    return [100 * float(lead[:, h - 1].mean()) for h in HORIZONS]


def waiting_numbers(p, base_ahead, high_ahead, high_gap):
    """How often the hedged portfolio is ahead after each horizon."""
    out = {"high_rate": p.high_rate}
    out["high_gap"] = 100 * high_gap
    for h, lo, hi in zip(HORIZONS, base_ahead, high_ahead):
        out[f"ahead_{h}"] = lo
        out[f"ahead_high_{h}"] = hi
    return out


def table(rows, header, widths):
    """A fixed-width text table, the first column left-aligned."""

    def line(cells):
        first, *rest = cells
        out = str(first).ljust(widths[0])
        for c, w in zip(rest, widths[1:], strict=True):
            out += "  " + str(c).rjust(w)
        return out

    return "\n".join([line(header)] + [line(row) for row in rows])


# the rows of the first table, as label, key and format
COMPOUND_ROWS = [
    ("average return, % a year", "average", "{:.2f}"),
    ("minus variance drag", "drag", "{:.2f}"),
    ("plus tail term", "tail", "{:.2f}"),
    ("growth rate, % a year", "growth", "{:.2f}"),
    ("volatility, % a year", "vol", "{:.1f}"),
    ("worst month, % loss", "worst", "{:.1f}"),
    ("median $1 after 30 years", "median_wealth", "${:.2f}"),
    ("5th percentile of $1", "p5_wealth", "${:.2f}"),
]


def compound_table(d):
    """Average and compounded return, unhedged and hedged."""
    cells = []
    for label, key, form in COMPOUND_ROWS:
        a = form.format(d[f"plain_{key}"])
        b = form.format(d[f"hedged_{key}"])
        cells.append([label, a, b])
    return table(cells, ["", "unhedged", "hedged"], [24, 8, 8])


def drawdown_table(d):
    """Thirty-year maximum drawdown of the three portfolios."""
    names = (("plain", "unhedged"), ("hedged", "hedged"))
    names += (("mix", f"{d['mix_exposure']:.0f}% stock"),)

    rows = []
    for key, label in names:
        cells = [d[f"{key}_dd_median"], d[f"{key}_dd_p95"]]
        cells += [d[f"{key}_dd_30"], d[f"{key}_dd_50"]]
        under = f"{d[f'{key}_under_years']:.1f}"
        rows.append([label, *(f"{c:.0f}%" for c in cells), under])

    header = ["", "median", "95th pct", "> 30%", "> 50%", "years under"]
    return table(rows, header, [9, 6, 8, 5, 5, 11])


def strike_table(prices):
    """What each strike costs and how often crashes must come."""
    rows = []
    for x in prices:
        cells = [f"{100 * x['strike']:.0f}%"]
        cells.append(f"{1200 * x['premium']:.2f}%")
        cells.append(f"{100 * x['vol']:.1f}")
        cells += [f"{x['arith']:.3f}", f"{x['geo']:.3f}"]
        rows.append(cells + [f"{1 / x['geo']:.1f}"])

    header = ["strike", "cost/yr", "vol", "arith.", "geom.", "years"]
    return table(rows, header, [6, 7, 4, 6, 5, 5])


def results(p, r):
    """Every number and table the article quotes, and the curves."""
    prices = pricing(p)
    parts = {"plain": growth_parts(p, 1.0, 0.0)}
    parts["hedged"] = growth_parts(p, 1.0, p.hedge)
    rates, gaps = sweep(p)

    # the simulation check sits at the grid rate nearest break-even
    geo_be = prices[STRIKES.index(p.strike)]["geo"]
    nearest = float(rates[np.argmin(np.abs(rates - geo_be))])
    check = (nearest, *at_rate(p, r["draws"], nearest))
    base_ahead = ahead(r["plain"], r["hedged"])
    high_ahead = ahead(*at_rate(p, r["draws"], p.high_rate))

    d = market_numbers(p, r, prices) | compound_numbers(parts, r)
    d |= path_numbers(p, r, parts) | drawdown_numbers(r)
    d |= break_even_numbers(p, prices, gaps, check)
    d |= exposure_numbers(p, r)
    high_gap = growth_gap(p, 1.0, p.high_rate)
    d |= waiting_numbers(p, base_ahead, high_ahead, high_gap)

    tables = {"compound": compound_table(d)}
    tables["drawdowns"] = drawdown_table(d)
    tables["strikes"] = strike_table(prices)
    curves = {"rates": rates, "gaps": gaps}
    curves |= {"ahead": base_ahead, "ahead_high": high_ahead}
    return {"derived": d, "tables": tables}, curves


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
    """Title, round y ticks, a zero line and a legend for a panel."""
    ax.set_title(title, pad=30 if legend else 12)

    lo, hi = ax.dataLim.intervaly
    steps = MaxNLocator(5, steps=[1, 2, 5, 10])
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
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def x_axis(ax, lo, hi, label, steps=6):
    """Round, evenly spaced x ticks that end on the axis limits."""
    ticks = MaxNLocator(steps, steps=[1, 2, 5, 10]).tick_values(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_xlabel(label)


def chart_path(p, r, out):
    """Wealth of $100 on the first path with crashes in at least two
    months of its first 15 years, with its crash months marked."""
    early = (r["crashes"][:, : 12 * 15] > 0).sum(axis=1)
    i = int(np.argmax(early >= 2))
    x = r["x"][i : i + 1]
    months = np.arange(12 * p.years + 1) / 12
    fig, ax = plt.subplots()

    lines = (("unhedged", 0.0, BLUE), ("hedged", p.hedge, ORANGE))
    for label, h, colour in lines:
        growth = np.log(portfolio_return(p, x, 1.0, h))[0]
        wealth = 100 * np.exp(np.r_[0, np.cumsum(growth)])
        ax.plot(months, wealth, color=colour, label=label)

    # a dotted line at the end of each month with a crash
    for m in np.flatnonzero(r["crashes"][i]):
        ax.axvline((m + 1) / 12, color=GREY, lw=0.8, ls=":")

    x_axis(ax, 0, p.years, "years")
    finish(ax, "Wealth of $100 on one simulated path", True, True)
    save(fig, out / "path.png")


def chart_drawdowns(r, out):
    """Share of paths whose deepest drawdown reaches each depth."""
    depth = np.arange(0, 81)
    fig, ax = plt.subplots()
    lines = (("plain", "unhedged", BLUE), ("hedged", "hedged", ORANGE))
    lines += (("mix", f"{100 * r['mix_exposure']:.0f}% stock", GREEN),)

    for key, label, colour in lines:
        deep = 100 * r[key]["deepest"]
        share = [100 * np.mean(deep >= d) for d in depth]
        ax.plot(depth, share, color=colour, label=label)

    x_axis(ax, 0, 80, "maximum drawdown over 30 years, in percent")
    title = "Share of paths with a drawdown this deep, in percent"
    finish(ax, title, legend=True)
    save(fig, out / "drawdowns.png")


def chart_break_even(p, curves, out):
    """Hedged minus unhedged growth against the real crash rate."""
    rates, gaps = curves["rates"], curves["gaps"]
    fig, ax = plt.subplots()
    colours = {0.6: BLUE, 1: ORANGE, 1.5: GREEN}
    for w, colour in colours.items():
        label = f"{100 * w:.0f}% stock"
        ax.plot(
            rates, 100 * np.array(gaps[w]), color=colour, label=label
        )

    # the real crash rate of the base case, and the priced one
    ax.axvline(p.crash_rate, color=GREY, lw=0.8, ls=":")
    ax.axvline(p.priced_rate, color=GREY, lw=0.8, ls="--")
    x_axis(ax, 0, 0.4, "real crashes a year")
    title = "Hedged minus unhedged growth, in percent a year"
    finish(ax, title, legend=True)
    save(fig, out / "break_even.png")


def chart_exposure(p, r, curves, out):
    """Growth and deep drawdowns against the equity exposure."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    lines = (("unhedged", 0.0, BLUE), ("hedged", p.hedge, ORANGE))

    for label, h, colour in lines:
        growth = [
            100 * growth_parts(p, w, h)["growth"] for w in EXPOSURES
        ]
        ax[0].plot(EXPOSURES, growth, color=colour, label=label)
        deep = [r["grid"][h, w]["deepest"] for w in EXPOSURES]
        p95 = [100 * np.percentile(d, 95) for d in deep]
        ax[1].plot(EXPOSURES, p95, color=colour)
    ax[1].axhline(100 * p.drawdown_limit, color=GREY, lw=0.8, ls="--")

    x_axis(
        ax[1], 0, 2, "equity exposure, as a multiple of the portfolio"
    )
    finish(ax[0], "Growth rate, in percent a year", legend=True)
    finish(ax[1], "95th percentile of the 30-year drawdown, in percent")
    save(fig, out / "exposure.png")


def chart_ahead(p, curves, out):
    """Share of paths where the hedged portfolio leads, by horizon."""
    fig, ax = plt.subplots()
    base = f"{p.crash_rate:.2f} crashes a year"
    high = f"{p.high_rate:.2f} crashes a year"
    lines = ((curves["ahead"], BLUE, base),)
    lines += ((curves["ahead_high"], ORANGE, high),)

    for share, colour, label in lines:
        ax.plot(HORIZONS, share, "o-", color=colour, label=label)

    x_axis(ax, 0, 35, "years since the start", steps=7)
    title = (
        "Share of paths where the hedged portfolio leads, in percent"
    )
    finish(ax, title, legend=True)
    save(fig, out / "ahead.png")


def main(out=Path(".")):
    p = BASE
    r = experiments(p)
    res, curves = results(p, r)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chart_path(p, r, figures)
    chart_drawdowns(r, figures)
    chart_break_even(p, curves, figures)
    chart_exposure(p, r, curves, figures)
    chart_ahead(p, curves, figures)

    text = json.dumps(res, indent=1) + "\n"
    (out / "results.json").write_text(text)
    for name in ("compound", "drawdowns", "strikes"):
        print(res["tables"][name], end="\n\n")


if __name__ == "__main__":
    main()
