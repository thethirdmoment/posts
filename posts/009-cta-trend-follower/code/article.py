# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

YEAR = 252  # trading days in a year
UP, DOWN, RANGE = 0, 1, 2  # the states of a sector's trend


def range_days(p):
    """Mean length of a sideways stretch, set so that the chain spends
    the chosen share of days in range in the long run."""
    s = p.range_share
    return s * p.trend_days / (p.to_range * (1 - s))


def trend_chain(p, u):
    """Each sector's trend state, day by day, from one uniform a day."""
    leave_trend = 1 / p.trend_days
    leave_range = 1 / range_days(p)
    state = np.empty(u.shape, dtype=np.int8)

    # the first day starts from the long-run shares of the states
    s = p.range_share
    first = np.where(u[0] < (1 - s) / 2, UP, DOWN)
    state[0] = np.where(u[0] >= 1 - s, RANGE, first)

    for t in range(1, len(u)):
        prev, x = state[t - 1], u[t]

        # a trend ends in a range or in the opposite trend
        ended = np.where(x < leave_trend * p.to_range, RANGE, 1 - prev)
        from_trend = np.where(x < leave_trend, ended, prev)

        # a range ends in an up or a down trend with equal chance
        ended = np.where(x < leave_range / 2, UP, DOWN)
        from_range = np.where(x < leave_range, ended, RANGE)

        state[t] = np.where(prev == RANGE, from_range, from_trend)
    return state


def daily_returns(p, draws, trend, storm):
    """Daily returns, shaped days by worlds by markets."""
    sigma = (
        calm_vols(p) * np.where(storm, p.storm_mult, 1.0)[..., SECTOR]
    )
    common = math.sqrt(p.sector_corr) * draws["factor"][..., SECTOR]
    r = sigma * (common + math.sqrt(1 - p.sector_corr) * draws["own"])

    # a trend drifts at a fixed Sharpe ratio, whatever the volatility
    state = trend[..., SECTOR]
    direction = (state == UP).astype(float) - (state == DOWN)
    r += direction * p.trend_sharpe / math.sqrt(YEAR) * sigma

    # a range pulls the price back to where the range began
    kappa = math.log(2) / p.half_life
    in_range = state == RANGE
    since_start = np.zeros(r.shape[1:])
    for t in range(len(r)):
        r[t] -= kappa * since_start * in_range[t]
        since_start = np.where(in_range[t], since_start + r[t], 0.0)
    return r


def trend_signal(r, lookback):
    """+1 after a rising lookback, -1 after a falling one, from the
    returns of the days before each day only."""
    before = np.cumsum(r, axis=0) - r
    past = np.zeros_like(r)
    past[lookback:] = before[lookback:] - before[:-lookback]
    return np.sign(past)


def ewma_variance(p, x, start):
    """Exponentially weighted average of squared values, using the
    values up to the day before each day."""
    keep = 1 - 1 / (p.vol_com + 1)
    v = np.empty_like(x)
    v[0] = start
    for t in range(1, len(x)):
        v[t] = keep * v[t - 1] + (1 - keep) * x[t - 1] ** 2
    return v


def positions(p, signal, var_hat, scheme):
    """Notional in each market, as a fraction of capital."""
    budget = p.target_vol / math.sqrt(MARKETS * YEAR)
    if scheme == "constant":
        return signal * budget / average_vols(p)
    return signal * budget / np.sqrt(var_hat)


def target_book(p, w, ret):
    """Scale the book so its estimated volatility is the target."""
    book = np.sum(w * ret, axis=2)
    var_hat = ewma_variance(p, book, p.target_vol**2 / YEAR)
    scale = p.target_vol / np.sqrt(YEAR * var_hat)
    return w * scale[..., None]


def market_pnl(p, w, ret):
    """Each market's daily P&L and trading cost, as fractions of
    capital. A position is set at one close and earns the next day."""
    traded = np.abs(np.diff(w, axis=0, prepend=0.0))
    cost = traded * p.cost_frac * calm_vols(p)
    return w * ret - cost, cost


def block_pnl(z, n):
    """P&L of holding the move so far in each block of n days, and the
    same P&L from the identity."""
    blocks = z[: len(z) // n * n].reshape(-1, n, *z.shape[1:])
    before = np.cumsum(blocks, axis=1) - blocks
    pnl = np.sum(before * blocks, axis=1)

    move = blocks.sum(axis=1)
    identity = 0.5 * (move**2 - np.sum(blocks**2, axis=1))
    return pnl, identity, move


@dataclass(frozen=True)
class Params:
    # the markets, four sectors of six, with calm volatility a year
    sector_vols: tuple = (0.16, 0.06, 0.09, 0.24)
    sector_corr: float = 0.5

    # the trend chain, in days
    trend_days: float = 250  # mean length of a trend
    to_range: float = 0.5  # chance a trend ends in a range
    range_share: float = 0.3  # long-run share of days in range
    trend_sharpe: float = 1.25  # of a trend's drift, a year
    half_life: float = 60  # of the pull back in a range

    # the volatility chain
    storm_mult: float = 2.5
    calm_days: float = 500
    storm_days: float = 60

    # the trend follower
    lookback: int = 126
    vol_com: float = 60  # centre of mass of the estimates, in days
    target_vol: float = 0.10
    cost_frac: float = 0.015  # of a calm day's move, per unit traded

    # the experiment
    worlds: int = 100
    years: int = 20
    burn_in: int = 252
    seed: int = 9


BASE = Params()
SECTORS, MARKETS = 4, 24
SECTOR = np.repeat(np.arange(SECTORS), MARKETS // SECTORS)
LOOKBACKS = (21, 63, 126, 252)
RANGE_SHARES = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75)
QUARTER = 63
SPIKE = 1.5  # a window this many times the average volatility


def calm_vols(p):
    """Calm daily volatility of each market."""
    per_year = np.repeat(p.sector_vols, MARKETS // SECTORS)
    return per_year / math.sqrt(YEAR)


def storm_share(p):
    """Long-run share of days in a storm."""
    return p.storm_days / (p.storm_days + p.calm_days)


def average_vols(p):
    """Root mean square daily volatility across calm and storm."""
    q = storm_share(p)
    return calm_vols(p) * math.sqrt(1 - q + q * p.storm_mult**2)


def storm_chain(p, u):
    """Each sector's volatility state, day by day, True in a storm."""
    storm = np.empty(u.shape, dtype=bool)
    storm[0] = u[0] < storm_share(p)
    for t in range(1, len(u)):
        leave = np.where(
            storm[t - 1], 1 / p.storm_days, 1 / p.calm_days
        )
        storm[t] = storm[t - 1] ^ (u[t] < leave)
    return storm


def trend_age(trend):
    """Days since each sector's current trend or range began."""
    age = np.zeros(trend.shape, dtype=np.int32)
    for t in range(1, len(trend)):
        same = trend[t] == trend[t - 1]
        age[t] = np.where(same, age[t - 1] + 1, 0)
    return age


def draw(p):
    """The uniforms of both chains and the normal shocks."""
    days = p.burn_in + YEAR * p.years
    sectors = (days, p.worlds, SECTORS)
    streams = np.random.SeedSequence(p.seed).spawn(4)
    trend, storm, factor, own = (
        np.random.default_rng(s) for s in streams
    )

    return {
        "trend": trend.random(sectors),
        "storm": storm.random(sectors),
        "factor": factor.standard_normal(sectors),
        "own": own.standard_normal((days, p.worlds, MARKETS)),
    }


def world(p, draws):
    """The regimes, returns and volatility estimates of every world."""
    trend = trend_chain(p, draws["trend"])
    storm = storm_chain(p, draws["storm"])
    r = daily_returns(p, draws, trend, storm)
    var_hat = ewma_variance(p, r, calm_vols(p) ** 2)
    return {"trend": trend, "storm": storm, "r": r, "var_hat": var_hat}


def run(p, wd, lookback=None, scheme="targeted", signal=None):
    """Each market's daily P&L and cost over the scored years."""
    if signal is None:
        signal = trend_signal(wd["r"], lookback or p.lookback)
    w = positions(p, signal, wd["var_hat"], scheme)
    if scheme == "targeted":
        w = target_book(p, w, wd["r"])
    pnl, cost = market_pnl(p, w, wd["r"])
    return {"pnl": pnl[p.burn_in :], "cost": cost[p.burn_in :]}


def book(x):
    """The whole book's daily return."""
    return x["pnl"].sum(axis=2)


def mean_se(values):
    """Mean and standard error over independent worlds."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def paired(a, b):
    """Mean and standard error of b minus a, world by world."""
    return mean_se(np.asarray(b) - np.asarray(a))


def sharpe(daily):
    """Annual Sharpe ratio of each world's daily returns."""
    return math.sqrt(YEAR) * daily.mean(0) / daily.std(0, ddof=1)


def blocks(x, n):
    """Sums over non-overlapping blocks of n days."""
    return x[: len(x) // n * n].reshape(-1, n, *x.shape[1:]).sum(1)


def skew(x):
    d = x - x.mean(0)
    return (d**3).mean(0) / (d**2).mean(0) ** 1.5


def kurtosis(x):
    d = x - x.mean(0)
    return (d**4).mean(0) / (d**2).mean(0) ** 2 - 3


def max_drawdown(daily):
    """Deepest fall of the summed returns, in annual volatilities."""
    total = np.cumsum(daily, axis=0)
    fall = np.max(np.maximum.accumulate(total, axis=0) - total, axis=0)
    return fall / (daily.std(0, ddof=1) * math.sqrt(YEAR))


def rolling_vol(daily, n=QUARTER):
    """Realized volatility a year over each window of n days."""
    sq = np.cumsum(np.vstack([np.zeros(daily.shape[1:]), daily**2]), 0)
    return np.sqrt((sq[n:] - sq[:-n]) / n * YEAR)


def labels(p, wd, lookback):
    """0 for an established trend, 1 for a new one, 2 for a range."""
    trend = wd["trend"][p.burn_in :]
    age = trend_age(wd["trend"])[p.burn_in :]
    label = np.where(age >= lookback, 0, 1)
    label = np.where(trend == RANGE, 2, label)
    return label[..., SECTOR]


def regime_split(p, x, label):
    """Share of market-days and P&L a year under each label."""
    gross = x["pnl"] + x["cost"]

    shares, parts = [], []
    for k in range(3):
        on = label == k
        shares.append(on.mean(axis=(0, 2)))
        parts.append(YEAR * np.sum(gross * on, axis=2).mean(0))
    parts.append(-YEAR * x["cost"].sum(axis=2).mean(0))
    return np.array(shares), np.array(parts)


def smile(p, daily, market, n=QUARTER):
    """Quadratic fit and quintile means of the book's n-day returns
    against the market's, standardized in each world."""
    y, x = 100 * blocks(daily, n), blocks(market, n)
    x = (x - x.mean(0)) / x.std(0)

    curve, quint, won = [], [], []
    for k in range(x.shape[1]):
        curve.append(np.polyfit(x[:, k], y[:, k], 2))

        # fifths of the market's quarters, and the worst tenth
        order = np.argsort(x[:, k])
        parts = np.array_split(order, 5)
        quint.append([(x[i, k].mean(), y[i, k].mean()) for i in parts])
        worst = order[: len(order) // 10]
        won.append(np.mean(y[worst, k] > 0))
    return np.array(curve), np.array(quint), np.array(won)


MOVE_EDGES = (0, 0.5, 1, 1.5, 2, np.inf)


def move_payoff(p, wd, x):
    """Per world, mean quarterly P&L in basis points by bin of the
    normalized move, and a fit on the move's size and log volatility."""
    r = wd["r"][p.burn_in :]
    z = r / np.sqrt(wd["var_hat"][p.burn_in :])
    move = np.abs(blocks(z, QUARTER)) / math.sqrt(QUARTER)
    vol = np.sqrt(blocks(r**2, QUARTER) / QUARTER) / calm_vols(p)
    pnl = 1e4 * blocks(x["pnl"], QUARTER)

    bins, fits = [], []
    for k in range(move.shape[1]):
        m, v, y = (
            move[:, k].ravel(),
            vol[:, k].ravel(),
            pnl[:, k].ravel(),
        )
        which = np.digitize(m, MOVE_EDGES) - 1
        bins.append([y[which == b].mean() for b in range(5)])
        design = np.column_stack([np.ones_like(m), m, np.log(v)])
        fits.append(np.linalg.lstsq(design, y, rcond=None)[0])
    return np.array(bins), np.array(fits)


def identity_check(p, wd):
    """The linear rule's quarterly P&L on the first market."""
    r = wd["r"][p.burn_in :, :, 0]
    z = r / np.sqrt(wd["var_hat"][p.burn_in :, :, 0])
    pnl, identity, move = block_pnl(z, QUARTER)
    in_range = blocks(wd["trend"][p.burn_in :, :, 0] == RANGE, QUARTER)
    return {
        "pnl": pnl,
        "move": move,
        "error": float(np.max(np.abs(pnl - identity))),
        "range": in_range > QUARTER / 2,
    }


def risk_numbers(daily):
    """Per world: volatility of volatility, share of windows above 1.5
    times the average, kurtosis, worst day and drawdown."""
    vol = rolling_vol(daily)
    return {
        "volvol": vol.std(0) / vol.mean(0),
        "spikes": np.mean(vol > SPIKE * vol.mean(0), axis=0),
        "kurt": kurtosis(daily),
        "worst": -daily.min(0) / daily.std(0),
        "drawdown": max_drawdown(daily),
    }


def experiments(p=BASE):
    draws = draw(p)
    wd = world(p, draws)
    out = {"world": wd, "draws": draws}
    out["schemes"] = {}
    for scheme in ("constant", "per_market", "targeted"):
        out["schemes"][scheme] = run(p, wd, scheme=scheme)
    base = out["schemes"]["targeted"]
    out["base"] = base
    out["label"] = labels(p, wd, p.lookback)
    out["identity"] = identity_check(p, wd)
    out["moves"] = move_payoff(p, wd, base)

    # an always-long book, and a world where the regimes do nothing
    ones = np.ones_like(wd["r"])
    out["long"] = sharpe(
        book(run(p, wd, scheme="per_market", signal=ones))
    )
    flat = replace(p, trend_sharpe=0.0, half_life=math.inf)
    x = run(flat, world(flat, draws))
    out["random"] = sharpe(book(x) + x["cost"].sum(2))
    out["sweep"], out["splits"] = sweep(p, draws)
    return out


def sweep(p, draws):
    grid, splits = {}, {}
    for s in RANGE_SHARES:
        q = replace(p, range_share=s)
        wd = world(q, draws)

        for lookback in LOOKBACKS:
            x = run(q, wd, lookback)
            grid[s, lookback] = sharpe(book(x))
            if s == p.range_share:
                label = labels(q, wd, lookback)
                splits[lookback] = regime_split(q, x, label)
        del wd
    return grid, splits


def setting_numbers(p):
    d = {"worlds": p.worlds, "years": p.years, "lookback": p.lookback}
    d |= {"trend_days": p.trend_days, "range_days": range_days(p)}
    d |= {"range_target": 100 * p.range_share}
    d |= {"storm_share": 100 * storm_share(p)}
    d |= {"trend_sharpe": p.trend_sharpe, "half_life": p.half_life}
    d |= {"storm_mult": p.storm_mult, "target": 100 * p.target_vol}

    # each sector's calm volatility, in percent a year
    names = ("equity", "bond", "currency", "commodity")
    for name, vol in zip(names, p.sector_vols):
        d[f"vol_{name}"] = 100 * vol

    d |= {"sector_corr": p.sector_corr, "vol_com": p.vol_com}
    d |= {"cost_pct": 100 * p.cost_frac, "quarter": QUARTER}
    d["cost_bp_equity"] = 1e4 * p.cost_frac * calm_vols(p)[0]

    # the ends of the two grids of the sweep
    d["short_lookback"], d["long_lookback"] = (
        LOOKBACKS[0],
        LOOKBACKS[-1],
    )
    d["least_range"] = 100 * RANGE_SHARES[0]
    d["most_range"] = 100 * RANGE_SHARES[-1]
    return d


def base_numbers(p, e):
    x = e["base"]
    daily = book(x)
    d = {}
    d["sharpe"], d["sharpe_se"] = mean_se(sharpe(daily))
    gross = sharpe(daily + x["cost"].sum(2))
    d["cost_drag"], d["cost_drag_se"] = paired(sharpe(daily), gross)
    d["realized_vol"] = 100 * float(np.mean(daily.std(0) * YEAR**0.5))
    d["mean_return"] = 100 * YEAR * float(daily.mean())

    d["skew"], d["skew_se"] = mean_se(skew(blocks(daily, 21)))
    d["dd_median"] = float(np.median(max_drawdown(daily)))
    d["losing_years"] = 100 * float(np.mean(blocks(daily, YEAR) < 0))

    # one market, one sector and the whole book, all net of costs
    market = sharpe(x["pnl"].reshape(-1, p.worlds * MARKETS))
    market = market.reshape(p.worlds, MARKETS).mean(1)
    by_sector = blocks(x["pnl"].transpose(2, 0, 1), 6).transpose(
        1, 2, 0
    )
    sector = np.mean([sharpe(by_sector[..., k]) for k in range(4)], 0)

    d["market_sharpe"], d["market_se"] = mean_se(market)
    d["sector_sharpe"], d["sector_se"] = mean_se(sector)
    d["sector_gain"], d["sector_gain_se"] = paired(market, sector)
    d["book_gain"], d["book_gain_se"] = paired(sector, sharpe(daily))
    return d


def regime_numbers(p, e):
    shares, parts = regime_split(p, e["base"], e["label"])
    d = {}
    for k, key in enumerate(("established", "new", "range")):
        d[f"{key}_share"] = 100 * float(shares[k].mean())
        d[key], d[f"{key}_se"] = mean_se(100 * parts[k])
    d["costs"], d["costs_se"] = mean_se(-100 * parts[3])

    d["long"], d["long_se"] = mean_se(e["long"])
    d["random"], d["random_se"] = mean_se(e["random"])

    # the linear rule's quarters, split by the regime they spent most in
    ident = e["identity"]
    d["identity_error"] = ident["error"]
    trend_mean, range_mean = [], []
    for k in range(p.worlds):
        in_range = ident["range"][:, k]
        trend_mean.append(ident["pnl"][~in_range, k].mean())
        range_mean.append(ident["pnl"][in_range, k].mean())

    d["ident_trend"], d["ident_trend_se"] = mean_se(trend_mean)
    d["ident_range"], d["ident_range_se"] = mean_se(range_mean)
    return d


def payoff_numbers(p, e):
    x, wd = e["base"], e["world"]
    market = wd["r"][p.burn_in :, :, :6].mean(2)
    equity = x["pnl"][..., :6].sum(2)

    d, fits = {}, {}
    for name, daily in (("book", book(x)), ("sleeve", equity)):
        curve, quint, won = smile(p, daily, market)
        fits[name] = (curve.mean(0), quint.mean(0))

        d[f"{name}_c"], d[f"{name}_c_se"] = mean_se(curve[:, 0])
        d[f"{name}_worst"], _ = mean_se(quint[:, 0, 1])
        d[f"{name}_middle"], _ = mean_se(quint[:, 1:4, 1].mean(1))
        d[f"{name}_best"], _ = mean_se(quint[:, 4, 1])
        d[f"{name}_won"] = 100 * float(won.mean())

    # the curvature at three horizons, also per month of horizon
    d["c_63"] = d["book_c"]
    for n in (21, 126):
        curve = smile(p, book(x), market, n)[0]
        d[f"c_{n}"], d[f"c_{n}_se"] = mean_se(curve[:, 0])
    for n in (21, 63, 126):
        d[f"c_{n}_month"] = d[f"c_{n}"] / (n / 21)

    bins, coef = e["moves"]
    for b in range(5):
        d[f"move_bin_{b}"] = float(bins[:, b].mean())
    d["move_coef"], d["move_coef_se"] = mean_se(coef[:, 1])
    d["vol_coef"], d["vol_coef_se"] = mean_se(coef[:, 2])
    return d, fits


def break_even(curve):
    """Range share where a falling Sharpe curve first crosses zero."""
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        if a > 0 >= b:
            s0, s1 = RANGE_SHARES[i], RANGE_SHARES[i + 1]
            return 100 * (s0 + (s1 - s0) * a / (a - b))

    return None


def sweep_numbers(p, e):
    grid, d = e["sweep"], {}
    for lookback in LOOKBACKS:
        curve = [grid[s, lookback].mean() for s in RANGE_SHARES]
        d[f"be_{lookback}"] = break_even(curve)
        for s in RANGE_SHARES:
            key = f"sr_{lookback}_{round(100 * s)}"
            d[key], d[key + "_se"] = mean_se(grid[s, lookback])

    # each lookback against the base case, and each step in range share
    base = grid[p.range_share, p.lookback]
    for lookback in LOOKBACKS:
        key = f"gain_{lookback}"
        d[key], d[key + "_se"] = paired(
            base, grid[p.range_share, lookback]
        )

    for a, b in zip(RANGE_SHARES, RANGE_SHARES[1:]):
        key = f"step_{round(100 * a)}_{round(100 * b)}"
        d[key], d[key + "_se"] = paired(
            grid[b, p.lookback], grid[a, p.lookback]
        )
    return d


def scheme_numbers(p, e):
    d, sharpes = {}, {}
    for name, x in e["schemes"].items():
        daily = book(x)
        sharpes[name] = sharpe(daily)
        d[f"{name}_sharpe"], d[f"{name}_sharpe_se"] = mean_se(
            sharpes[name]
        )

        for key, values in risk_numbers(daily).items():
            d[f"{name}_{key}"] = float(np.mean(values))
        d[f"{name}_drawdown"] = float(np.median(max_drawdown(daily)))

    pairs = (("constant", "per_market"), ("per_market", "targeted"))
    for a, b in pairs:
        key = f"gain_{a}_{b}"
        d[key], d[key + "_se"] = paired(sharpes[a], sharpes[b])
    return d


def table(rows, header, widths):
    """A fixed-width text table, the first column left-aligned."""

    def line(cells):
        first, *rest = cells
        out = str(first).ljust(widths[0])
        for c, w in zip(rest, widths[1:], strict=True):
            out += "  " + str(c).rjust(w)
        return out

    return "\n".join([line(header)] + [line(row) for row in rows])


REGIME_ROWS = (
    ("established trend", "established"),
    ("new trend", "new"),
    ("range", "range"),
)


def regime_table(d):
    rows = []
    for label, key in REGIME_ROWS:
        rows.append(
            [label, f"{d[key + '_share']:.0f}%", f"{d[key]:.1f}%"]
        )
    rows.append(["costs", "", f"{-d['costs']:.1f}%"])
    rows.append(["total", "100%", f"{d['mean_return']:.1f}%"])
    return table(rows, ["", "days", "return a year"], [17, 4, 13])


def lookback_table(p, e):
    rows = []
    for lookback in LOOKBACKS:
        shares, parts = e["splits"][lookback]
        cells = [f"{100 * v.mean():.1f}%" for v in parts]
        sr = e["sweep"][p.range_share, lookback].mean()
        rows.append([f"{lookback} days", *cells, f"{sr:.2f}"])
    header = ["lookback", "trend", "new", "range", "costs", "Sharpe"]
    return table(rows, header, [8, 5, 6, 6, 6, 6])


def move_table(d):
    names = ("0 to 0.5", "0.5 to 1", "1 to 1.5", "1.5 to 2", "above 2")
    rows = []
    for b, name in enumerate(names):
        rows.append([name, f"{d[f'move_bin_{b}']:.1f}"])

    return table(rows, ["size of move", "P&L, bp"], [12, 7])


SCHEMES = (
    ("constant", "constant notional"),
    ("per_market", "per market"),
    ("targeted", "plus whole book"),
)


SCHEME_HEADER = ("", "Sharpe", "vol of vol", "> 1.5x", "kurt", "worst")


def scheme_table(d):
    rows = []
    for key, label in SCHEMES:
        cells = [f"{d[key + '_sharpe']:.2f}"]
        cells.append(f"{d[key + '_volvol']:.2f}")
        cells.append(f"{100 * d[key + '_spikes']:.1f}%")

        cells += [f"{d[key + '_kurt']:.1f}", f"{d[key + '_worst']:.1f}"]
        rows.append([label, *cells, f"{d[key + '_drawdown']:.1f}"])

    header = [*SCHEME_HEADER, "dd"]
    return table(rows, header, [17, 6, 10, 6, 4, 5, 3])


# the text states these as losses or falls, so it needs their size
LOSSES = ("range", "vol_coef", "gain_21", "sr_126_75", "ident_range")
LOSSES += ("gain_per_market_targeted",)


def results(p, e):
    d = setting_numbers(p) | base_numbers(p, e) | regime_numbers(p, e)
    payoff, fits = payoff_numbers(p, e)
    d |= payoff | sweep_numbers(p, e) | scheme_numbers(p, e)

    for key in LOSSES:
        d[key + "_abs"] = abs(d[key])
    d["identity_digits"] = math.floor(-math.log10(d["identity_error"]))
    d["spikes_pct"] = 100 * d["constant_spikes"]
    d["spike_level"] = SPIKE
    for part in ("worst", "middle", "best"):
        d[f"other_{part}"] = d[f"book_{part}"] - d[f"sleeve_{part}"]
    scaling = d["gain_per_market_targeted"]
    d["book_scaling_t"] = (
        abs(scaling) / d["gain_per_market_targeted_se"]
    )

    tables = {"regimes": regime_table(d), "moves": move_table(d)}
    tables["lookbacks"] = lookback_table(p, e)
    tables["schemes"] = scheme_table(d)
    return {"derived": d, "tables": tables}, fits


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


def chart_identity(e, out):
    ident = e["identity"]
    move, pnl = ident["move"].ravel(), ident["pnl"].ravel()
    fig, ax = plt.subplots()
    ax.scatter(move, pnl, s=6, color=BLUE, alpha=0.25, lw=0)

    # the identity with the squared daily moves at their average
    edge = 10 * math.ceil(np.abs(move).max() / 10)
    grid = np.linspace(-edge, edge, 200)
    ax.plot(grid, 0.5 * (grid**2 - QUARTER), color=GREY, lw=1)

    label = "the quarter's total move, in daily volatilities"
    x_axis(ax, -edge, edge, label)
    name = "Linear rule's quarterly P&L, in squared daily volatilities"
    finish(ax, name)
    save(fig, out / "identity.png")


def chart_example(p, e, out):
    wd, x = e["world"], e["base"]
    years = np.arange(1, YEAR * p.years + 1) / YEAR
    market = wd["r"][p.burn_in :, 0, :6].mean(1)
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    ax[0].plot(years, 100 * np.cumsum(market), color=BLUE)
    ax[1].plot(years, 100 * np.cumsum(book(x)[:, 0]), color=ORANGE)
    x_axis(ax[1], 0, p.years, "years")
    finish(ax[0], "Stock market, cumulative return in percent")
    finish(ax[1], "Trend follower, cumulative return in percent")
    save(fig, out / "example.png")


def chart_smile(fits, out):
    fig, ax = plt.subplots()
    grid = np.linspace(-2, 2, 100)
    lines = (
        ("book", "whole book", BLUE),
        ("sleeve", "stock indices", ORANGE),
    )

    for key, label, colour in lines:
        curve, quint = fits[key]
        fitted = np.polyval(curve, grid)
        ax.plot(grid, fitted, color=colour, label=label)
        ax.plot(quint[:, 0], quint[:, 1], "o", color=colour)

    x_axis(ax, -2, 2, "stock market's quarter, in standard deviations")
    title = "Trend follower's quarterly return, in percent"
    finish(ax, title, legend=True)
    save(fig, out / "smile.png")


def chart_sweep(e, out):
    fig, ax = plt.subplots()
    shares = 100 * np.array(RANGE_SHARES)
    for lookback, colour in zip((21, 126, 252), (BLUE, ORANGE, GREEN)):
        sr = [e["sweep"][s, lookback].mean() for s in RANGE_SHARES]
        label = f"{lookback}-day lookback"
        ax.plot(shares, sr, "o-", color=colour, label=label)
    x_axis(ax, 0, 80, "share of days in range, in percent")
    finish(ax, "Net Sharpe ratio of the whole book", legend=True)
    save(fig, out / "sweep.png")


def chart_rolling(p, e, out):
    fig, ax = plt.subplots()
    lines = (("constant", "constant notional", BLUE),)
    lines += (("targeted", "fully targeted", ORANGE),)

    for key, label, colour in lines:
        vol = rolling_vol(book(e["schemes"][key]))[:, 0]
        vol = 10 * vol / vol.mean()
        days = np.arange(QUARTER, QUARTER + len(vol)) / YEAR
        ax.plot(days, vol, color=colour, label=label)
    x_axis(ax, 0, p.years, "years")
    finish(
        ax, "Volatility over the last 63 days, in percent", legend=True
    )
    save(fig, out / "rolling.png")


def main(out=Path(".")):
    p = BASE
    e = experiments(p)
    res, fits = results(p, e)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chart_identity(e, figures)
    chart_example(p, e, figures)
    chart_smile(fits, figures)
    chart_sweep(e, figures)
    chart_rolling(p, e, figures)

    text = json.dumps(res, indent=1) + "\n"
    (out / "results.json").write_text(text)
    for name in res["tables"]:
        print(res["tables"][name], end="\n\n")


if __name__ == "__main__":
    main()
