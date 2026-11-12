# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


def volume_profile(p):
    """Expected market volume in each slice of the day, in shares.

    The curve is U-shaped, about three times as high at the open and
    the close as at midday, and it sums to the average daily volume.
    """
    t = (np.arange(p.slices) + 0.5) / p.slices
    weight = 1 + 2 * (2 * t - 1) ** 2
    return p.adv * weight / weight.sum()


def fill_prices(p, n, shocks):
    """The price paid in each slice of each day, and the price each
    slice starts from. n holds the shares bought in each slice and
    shocks the standard normal price shocks, one row per day."""
    tau = 1 / p.slices
    volume = volume_profile(p)

    # each slice moves the price by its shock, the drift and the
    # permanent impact of the shares bought in it
    step = p.sigma * math.sqrt(tau) * shocks + p.drift * tau
    step = step + p.gamma * n
    start = p.s0 + np.cumsum(step, axis=1) - step

    # the buyer pays half the spread and the temporary impact on top
    temporary = p.eta * n / volume
    return start + p.half_spread + temporary, start


def cost_parts(p, n):
    """Expected shortfall of a schedule split into its parts, in
    dollars, and the variance of its timing part."""
    tau = 1 / p.slices
    remaining = p.shares - np.cumsum(n)  # still to buy after a slice
    temporary = np.sum(p.eta * n**2 / volume_profile(p))
    permanent = p.gamma * (p.shares**2 - np.sum(n**2)) / 2

    return {
        "fixed": p.half_spread * p.shares,
        "temporary": float(temporary),
        "permanent": float(permanent),
        "drift": p.drift * tau * float(remaining.sum()),
        "variance": p.sigma**2 * tau * float(np.sum(remaining**2)),
    }


def twap(p):
    """The same number of shares in every slice."""
    return np.full(p.slices, p.shares / p.slices)


def vwap(p):
    """Shares in proportion to the market's expected volume."""
    volume = volume_profile(p)
    return p.shares * volume / volume.sum()


def optimal(p, urgency, drift=0.0):
    """The schedule that minimises expected shortfall plus urgency
    times its variance, when the price drifts by `drift` a day.

    The unknowns are the shares still to buy after each slice, and
    the first-order conditions are one tridiagonal linear system.
    """
    tau = 1 / p.slices
    a = p.eta / volume_profile(p) - p.gamma / 2
    risk = urgency * p.sigma**2 * tau

    # row j is the condition on the shares left after slice j + 1
    matrix = np.diag(a[:-1] + a[1:] + risk)
    matrix -= np.diag(a[1:-1], 1) + np.diag(a[1:-1], -1)
    rhs = np.full(p.slices - 1, -drift * tau / 2)
    rhs[0] += a[0] * p.shares

    remaining = np.linalg.solve(matrix, rhs)
    held = np.r_[p.shares, remaining, 0.0]
    return held[:-1] - held[1:]


HOURS = 6.5  # one trading session


def urgency(p, half_life_hours):
    """The urgency whose schedule has the given half-life, from the
    continuous-time kappa at the day's average volume."""
    half_life = half_life_hours / HOURS  # in trading days
    eta_bar = p.eta / p.adv
    return eta_bar / (p.sigma**2 * half_life**2)


def vwap_slippage(p, n, volumes, prices, start):
    """Our average price minus the day's market VWAP, times the shares.
    The rest of the market trades at each slice's starting price, and
    our own fills are part of the VWAP."""
    market_value = np.sum(volumes * start, axis=1) + prices @ n
    market_volume = volumes.sum(axis=1) + p.shares
    day_vwap = market_value / market_volume
    return prices @ n - p.shares * day_vwap


@dataclass(frozen=True)
class Params:
    # the order, bought between the open and the close
    shares: float = 1_000_000
    s0: float = 50.0  # the arrival price, in dollars

    # the market
    slices: int = 78  # five-minute slices in a 6.5-hour session
    sigma: float = 1.0  # dollars per share per square root of a day
    adv: float = 10_000_000  # average daily volume, in shares
    day_noise: float = 0.25  # log-sd of the day's volume factor
    slice_noise: float = 0.4  # log-sd of each slice's volume factor
    drift: float = 0.0  # dollars per share per day
    signal: float = 0.15  # the drift the manager expects

    # the costs of trading
    half_spread: float = 0.01  # dollars per share
    eta: float = 0.50  # dollars per share at 100% participation
    gamma: float = 1e-7  # permanent rise per share bought, in dollars

    # the experiment
    days: int = 20_000
    seed: int = 7


BASE = Params()
NOTIONAL = BASE.shares * BASE.s0


def draw_market(p):
    """Price shocks and the rest of the market's volume, one row per
    simulated day."""
    seeds = np.random.SeedSequence(p.seed).spawn(2)
    rng_price, rng_volume = (np.random.default_rng(s) for s in seeds)
    shocks = rng_price.standard_normal((p.days, p.slices))

    # lognormal day and slice factors, each with a mean of one
    def factor(sd, shape):
        z = rng_volume.standard_normal(shape)
        return np.exp(sd * z - sd**2 / 2)

    day = factor(p.day_noise, (p.days, 1))
    noise = factor(p.slice_noise, (p.days, p.slices))
    return shocks, volume_profile(p) * day * noise


def run(p, n, market):
    """Every simulated day of one schedule, as arrays over days."""
    shocks, volumes = market
    prices, start = fill_prices(p, n, shocks)
    remaining = p.shares - np.cumsum(n)
    unaffected = p.sigma * math.sqrt(1 / p.slices) * shocks

    return {
        "n": n,
        "parts": cost_parts(p, n),
        "shortfall": prices @ n - p.shares * p.s0,
        "timing": unaffected @ remaining,
        "move": unaffected.sum(axis=1),
        "slippage": vwap_slippage(p, n, volumes, prices, start),
        "participation": float(np.max(n / volume_profile(p))),
    }


def bp(dollars):
    """Dollars as basis points of the order's arrival value."""
    return 1e4 * np.asarray(dollars) / NOTIONAL


def paired(a, b, key="shortfall", days=slice(None)):
    """Mean and standard error of b minus a, in bp, day by day."""
    d = bp(b[key] - a[key])[days]
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def first_hour_pct(p, n):
    """Percent of the order bought in the first hour."""
    return 100 * float(n[: round(p.slices / HOURS)].sum()) / p.shares


def summary(r):
    """Mean, spread and closed form of one schedule's shortfall."""
    parts = r["parts"]
    shown = ("fixed", "temporary", "permanent", "drift")
    out = {x: float(bp(parts[x])) for x in shown}
    shortfall, timing = bp(r["shortfall"]), bp(r["timing"])
    root = math.sqrt(len(shortfall))

    return out | {
        "mean": float(shortfall.mean()),
        "sd": float(shortfall.std(ddof=1)),
        "p95": float(np.percentile(shortfall, 95)),
        "timing_mean": float(timing.mean()),
        "timing_se": float(timing.std(ddof=1) / root),
        "cf_mean": sum(out.values()),
        "cf_sd": float(bp(math.sqrt(parts["variance"]))),
        "participation": 100 * r["participation"],
        "first_hour": first_hour_pct(BASE, r["n"]),
    }


HALF_LIVES = (4, 2, 1)  # hours, the risk-averse schedules shown
FRONTIER = tuple(8 * 2 ** (-k / 3) for k in range(16))  # 8 h to 15 min


def schedules(p):
    """TWAP, VWAP and the three risk-averse schedules."""
    out = {"twap": twap(p), "vwap": vwap(p)}
    for h in HALF_LIVES:
        out[f"h{h}"] = optimal(p, urgency(p, h))
    return out


def drift_schedules(p):
    """The schedules compared when the price drifts up."""
    lam = urgency(p, 2)
    return {
        "twap": twap(p),
        "vwap": vwap(p),
        "blind": optimal(p, lam),
        "neutral": optimal(p, 0.0, p.signal),
        "aware": optimal(p, lam, p.signal),
    }


def experiments(p=BASE):
    """Every run the article compares."""
    market = draw_market(p)
    rising = replace(p, drift=p.signal)
    base = {k: run(p, n, market) for k, n in schedules(p).items()}
    frontier = [optimal(p, urgency(p, h)) for h in FRONTIER]
    drift = drift_schedules(p)

    return {
        "base": base,
        "neutral": run(p, optimal(p, 0.0), market),
        "frontier": [run(p, n, market) for n in frontier],
        "drift": {k: run(rising, n, market) for k, n in drift.items()},
        "flat": {k: run(p, n, market) for k, n in drift.items()},
    }


def frontier_curve(p, points=400):
    """Expected shortfall and its standard deviation in bp, from the
    risk-neutral schedule to a 6-minute half-life."""
    half_lives = np.geomspace(100, 0.1, points)
    lams = [0.0] + [urgency(p, h) for h in half_lives]

    means, sds = [], []
    for lam in lams:
        parts = cost_parts(p, optimal(p, lam))
        cost = parts["fixed"] + parts["temporary"] + parts["permanent"]
        means.append(float(bp(cost)))
        sds.append(float(bp(math.sqrt(parts["variance"]))))
    return np.array(means), np.array(sds)


def order_numbers(s):
    """Numbers for the sections on the order and the market."""
    p, volume = BASE, volume_profile(BASE)
    v = s["vwap"]
    out = {"notional": NOTIONAL, "bp_dollars": NOTIONAL / 1e4}
    out["days"] = p.days
    out["daily_vol_pct"] = 100 * p.sigma / p.s0
    out["open_to_midday"] = float(volume[0] / volume.min())
    out["share_of_day_pct"] = 100 * p.shares / (p.adv + p.shares)
    out["temp_10pct_cents"] = 100 * p.eta * 0.10
    out["fixed_cents"] = 100 * p.half_spread
    out["perm_total_cents"] = 100 * p.gamma * p.shares

    gap = np.abs(optimal(p, 0.0) - vwap(p)).max()
    out["neutral_vs_vwap_shares"] = float(gap)
    for key in ("twap", "vwap", "h2"):
        out[f"{key}_first_hour_pct"] = s[key]["first_hour"]

    fixed_perm = v["fixed"] + v["permanent"]
    out["vwap_fixed_perm_share_pct"] = 100 * fixed_perm / v["cf_mean"]
    out["vwap_sd_over_mean"] = v["sd"] / v["mean"]
    out["vwap_mean_usd"] = v["mean"] * NOTIONAL / 1e4
    return out


def schedule_numbers(r, s):
    """Paired differences against VWAP, and the zero checks."""
    base, out = r["base"], {}
    for key in ("twap", "h4", "h2", "h1"):
        diff = paired(base["vwap"], base[key])
        out[f"{key}_vs_vwap"], out[f"{key}_vs_vwap_se"] = diff
    out["twap_minus_vwap_usd"] = out["twap_vs_vwap"] * NOTIONAL / 1e4

    perms = [x["permanent"] for x in s.values()]
    out["perm_min"], out["perm_max"] = min(perms), max(perms)
    z = [abs(x["timing_mean"]) / x["timing_se"] for x in s.values()]
    out["timing_max_z"] = max(z)
    gaps = [100 * abs(x["sd"] / x["cf_sd"] - 1) for x in s.values()]
    out["sd_gap_max_pct"] = max(gaps)

    # TWAP's participation in the quietest slice, against VWAP's
    slice_shares = BASE.shares / BASE.slices
    quiet = volume_profile(BASE).min()
    out["twap_midday_part_pct"] = 100 * slice_shares / quiet
    out["vwap_part_pct"] = s["vwap"]["participation"]
    return out


def frontier_numbers(r, s, curve):
    """The cost of cutting risk, and TWAP against the frontier."""
    means, sds = curve
    rn = summary(r["neutral"])
    cf = [summary(x) for x in r["frontier"]]

    def cost_at(sd):
        return float(np.interp(sd, sds[::-1], means[::-1])) - means[0]

    def saved_per_bp(a, b):
        return (a["cf_sd"] - b["cf_sd"]) / (b["cf_mean"] - a["cf_mean"])

    out = {"quarter_cut_cost": cost_at(0.75 * sds[0])}
    out["half_cut_cost"] = cost_at(0.5 * sds[0])
    first, last = saved_per_bp(rn, cf[0]), saved_per_bp(cf[9], cf[12])
    out["slope_ratio"] = first / last
    out["twap_sd_minus_rn"] = s["twap"]["cf_sd"] - sds[0]
    twap_cost = paired(r["neutral"], r["base"]["twap"])
    out["twap_vs_neutral"], out["twap_vs_neutral_se"] = twap_cost
    out["fastest_participation"] = cf[-1]["participation"]
    return out


def benchmark_numbers(r):
    """Slippage against VWAP for VWAP and the 2-hour schedule."""
    base, out = r["base"], {}

    for key in ("vwap", "h2"):
        slip = bp(base[key]["slippage"])
        shortfall = bp(base[key]["shortfall"])
        out[f"{key}_slip_mean"] = float(slip.mean())
        out[f"{key}_slip_sd"] = float(slip.std(ddof=1))
        hidden = 1 - slip.mean() / shortfall.mean()
        out[f"{key}_hidden_pct"] = 100 * float(hidden)
        out[f"{key}_corr"] = float(np.corrcoef(slip, shortfall)[0, 1])
    out["h2_negative_corr"] = -out["h2_corr"]

    # the 2-hour schedule against VWAP, on rising and falling days
    up = base["vwap"]["move"] > 0
    gain = paired(base["h2"], base["vwap"], "slippage", up)
    loss = paired(base["vwap"], base["h2"], "slippage", ~up)
    out["slip_up_gain"], out["slip_up_gain_se"] = gain
    out["slip_down_loss"], out["slip_down_loss_se"] = loss
    return out


def drift_numbers(r):
    """Numbers for the section on a rising price."""
    d, f = r["drift"], r["flat"]
    out = {"signal_cents": 100 * BASE.signal}
    out["signal_bp"] = float(bp(BASE.signal * BASE.shares))
    out["drift_twap_extra"] = paired(f["twap"], d["twap"])[0]

    # savings against TWAP with the drift, and costs without it
    for key in ("blind", "neutral", "aware"):
        out[f"{key}_first_hour_pct"] = first_hour_pct(BASE, d[key]["n"])
        saved, saved_se = paired(d[key], d["twap"])
        out[f"drift_{key}_vs_twap"] = saved
        out[f"drift_{key}_vs_twap_se"] = saved_se
        wrong, wrong_se = paired(f["twap"], f[key])
        out[f"wrong_{key}_vs_twap"] = wrong
        out[f"wrong_{key}_vs_twap_se"] = wrong_se
    return out


NAMES = {
    "twap": "TWAP",
    "vwap": "VWAP",
    "h4": "4-hour",
    "h2": "2-hour",
    "h1": "1-hour",
    "blind": "2-hour, no drift",
    "neutral": "drift, neutral",
    "aware": "drift, 2-hour",
}


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def f1(values):
    return [f"{v:.1f}" for v in values]


def decomposition_table(s):
    """The four parts of the mean shortfall, in bp."""
    rows = []
    for key in ("twap", "vwap", "h2"):
        x = s[key]
        cells = [x["fixed"], x["permanent"], x["temporary"]]
        cells += [x["timing_mean"], x["mean"], x["sd"]]
        rows.append([NAMES[key], *f1(cells)])
    header = ["schedule", "fixed", "permanent", "temporary", "timing"]
    header += ["total", "SD"]
    return table(rows, header, [8, 5, 9, 9, 6, 5, 5])


def schedule_table(r, s):
    """Mean, SD, bad day and peak participation of each schedule."""
    rows = []
    for key, x in s.items():
        diff, se = paired(r["base"]["vwap"], r["base"][key])
        cells = f1([x["mean"], x["sd"], x["p95"], diff, se])
        rows.append([NAMES[key], *cells, f"{x['participation']:.0f}%"])
    header = ["schedule", "mean", "SD", "95th pct", "vs VWAP"]
    header += ["std err", "peak part."]
    return table(rows, header, [8, 4, 5, 8, 7, 7, 10])


def benchmark_table(r):
    """Shortfall against arrival and slippage against VWAP, in bp."""
    rows = []
    for key in ("vwap", "h2"):
        x = r["base"][key]
        shortfall, slip = bp(x["shortfall"]), bp(x["slippage"])
        cells = [shortfall.mean(), shortfall.std(ddof=1)]
        cells += [slip.mean(), slip.std(ddof=1)]

        corr = np.corrcoef(slip, shortfall)[0, 1]
        corr_cell = f"{corr:.2f}".replace("-0.00", "0.00")
        rows.append([NAMES[key], *f1(cells), corr_cell])
    header = ["schedule", "IS mean", "IS SD", "VWAP mean", "VWAP SD"]
    return table(rows, header + ["corr."], [8, 7, 5, 9, 7, 5])


def drift_table(r):
    """Each schedule with the drift, and on days without it."""
    d, f, rows = r["drift"], r["flat"], []
    for key in d:
        x, flat = summary(d[key]), summary(f[key])
        diff, se = paired(d[key], d["twap"])
        cells = f1([x["mean"], x["sd"], diff, se, flat["mean"]])
        rows.append([NAMES[key], f"{x['first_hour']:.0f}%", *cells])
    header = ["schedule", "1st hour", "mean", "SD", "saved"]
    header += ["std err", "no drift"]
    return table(rows, header, [16, 8, 4, 5, 5, 7, 8])


def results(r, curve):
    """Every number and table the article quotes."""
    s = {k: summary(x) for k, x in r["base"].items()}
    derived = order_numbers(s) | schedule_numbers(r, s)
    derived |= frontier_numbers(r, s, curve) | benchmark_numbers(r)
    derived |= drift_numbers(r)

    tables = {
        "decomposition": decomposition_table(s),
        "schedules": schedule_table(r, s),
        "benchmarks": benchmark_table(r),
        "drift": drift_table(r),
    }

    frontier = [summary(x) for x in r["frontier"]]
    out = {"schedules": s, "derived": derived, "tables": tables}
    out["frontier"] = [
        x | {"half_life": h} for x, h in zip(frontier, FRONTIER)
    ]
    out["drift"] = {k: summary(x) for k, x in r["drift"].items()}
    return out


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


def chart_schedules(out):
    """The share of the order still to buy through the day."""
    p, sched = BASE, schedules(BASE)
    hours = np.arange(p.slices + 1) * HOURS / p.slices
    plain = (("twap", BLUE, "TWAP"), ("vwap", ORANGE, "VWAP"))
    fast = (("h4", BLUE, "4-hour half-life"), ("h2", ORANGE, "2-hour"))
    fast += (("h1", GREEN, "1-hour"),)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)

    for a, lines in zip(ax, (plain, fast), strict=True):
        for key, colour, label in lines:
            bought = np.r_[0, np.cumsum(sched[key])] / p.shares
            a.plot(hours, 100 * (1 - bought), color=colour, label=label)
    ax[1].set_xticks(range(7))
    ax[1].set_xlabel("hours into the session")

    title = "Share of the order still to buy, in percent"
    finish(ax[0], title, legend=True)
    finish(ax[1], "Three Almgren-Chriss schedules", legend=True)
    save(fig, out / "schedules.png")


def chart_histogram(r, out):
    """Daily shortfall of VWAP and the 2-hour schedule, in bp."""
    pair = (("vwap", BLUE, "VWAP"), ("h2", ORANGE, "2-hour half-life"))
    days = [bp(r["base"][key]["shortfall"]) for key, _, _ in pair]
    lo = 50 * math.floor(min(x.min() for x in days) / 50)
    hi = 50 * math.ceil(max(x.max() for x in days) / 50)
    bins = np.arange(lo, hi + 1, 10)
    fig, ax = plt.subplots()

    for x, (_, colour, label) in zip(days, pair, strict=True):
        weights = np.full(len(x), 100 / len(x))
        ax.hist(x, bins, weights=weights, histtype="step", color=colour)
        ax.plot([], [], color=colour, label=label)
        ax.axvline(x.mean(), color=colour, lw=1, ls="--")
    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10]))
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        "implementation shortfall, in bp of the order's value"
    )
    finish(ax, "Share of days, in percent", legend=True)
    save(fig, out / "histogram.png")


def chart_frontier(r, curve, out):
    """Expected shortfall against its standard deviation, in bp."""
    means, sds = curve
    grid = [summary(x) for x in r["frontier"]]
    keep = sds >= min(x["cf_sd"] for x in grid) - 1e-9
    fig, ax = plt.subplots()

    # the line and the dots are closed forms, the dots at half-lives
    label = "efficient frontier"
    ax.plot(sds[keep], means[keep], color=BLUE, label=label)
    dots_sd = [x["cf_sd"] for x in grid]
    ax.plot(
        dots_sd, [x["cf_mean"] for x in grid], "o", color=BLUE, ms=4
    )

    marked = (("twap", ORANGE, "TWAP"), ("vwap", GREEN, "VWAP"))
    for key, colour, label in marked:
        x = summary(r["base"][key])
        ax.plot(
            x["cf_sd"], x["cf_mean"], "o", color=colour, label=label
        )

    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10]))
    ax.set_xlabel("standard deviation of the shortfall, in bp")
    title = "Expected shortfall against its risk, in bp"
    finish(ax, title, legend=True)
    save(fig, out / "frontier.png")


def chart_benchmarks(r, out):
    """Slippage against VWAP and shortfall against arrival, by day."""
    pair = (("vwap", BLUE, "VWAP"), ("h2", ORANGE, "2-hour half-life"))
    fig, ax = plt.subplots()
    for key, colour, label in pair:
        x = bp(r["base"][key]["slippage"][:2000])
        y = bp(r["base"][key]["shortfall"][:2000])
        ax.scatter(
            x, y, s=6, alpha=0.4, lw=0, color=colour, label=label
        )

    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10]))
    ax.set_xlabel("slippage against the day's VWAP, in bp")
    title = "Shortfall against the arrival price, in bp"
    finish(ax, title, legend=True)

    # legend dots opaque and larger than the faint scatter points
    for handle in ax.get_legend().legend_handles:
        handle.set_alpha(1)
        handle.set_sizes([30])
    save(fig, out / "benchmarks.png")


def main(out=Path(".")):
    r = experiments()
    curve = frontier_curve(BASE)
    res = results(r, curve)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chart_schedules(figures)
    chart_histogram(r, figures)
    chart_frontier(r, curve, figures)
    chart_benchmarks(r, figures)

    text = json.dumps(res, indent=1) + "\n"
    (out / "results.json").write_text(text)
    print(
        res["tables"]["schedules"], res["tables"]["drift"], sep="\n\n"
    )


if __name__ == "__main__":
    main()
