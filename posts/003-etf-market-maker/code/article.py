# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

BP = 1e-4  # one basis point


def basket(p):
    """Weight, beta and half-spread in bp of each stock in the fund.

    Weights fall with rank, and smaller holdings cost more to trade.
    """
    rank = np.arange(1, p.stocks + 1)
    weight = rank**-p.weight_decay
    weight = weight / weight.sum()
    beta = np.linspace(p.beta_low, p.beta_high, p.stocks)
    half_spread = np.linspace(p.spread_low, p.spread_high, p.stocks)
    return weight, beta, half_spread


def fair_value(p, factor, idio):
    """The fund's value per share at every minute, from the minute
    log returns of the market factor and of each stock's own news."""
    weight, beta, _ = basket(p)
    var = beta**2 * p.factor_vol**2 + p.idio_vol**2
    returns = np.outer(factor, beta) + idio - 0.5 * var * DT

    # the fund holds a fixed number of shares of each stock
    growth = np.exp(np.cumsum(returns, axis=0))
    return p.nav0 * np.r_[1.0, growth @ weight]


def quotes(p, fair, inventory):
    """Bid and offer around fair value, leaned against inventory."""
    units = inventory / p.unit
    held = max(-p.lean_cap, min(p.lean_cap, units))
    mid = fair * (1 - p.lean_bp * held * BP)
    half = max(p.tick, p.half_spread_bp * BP * fair)

    # round away from the leaned mid onto the cent grid
    bid = math.floor((mid - half) / p.tick + 1e-9) * p.tick
    ask = math.ceil((mid + half) / p.tick - 1e-9) * p.tick
    return bid, ask


def chance(p, distance_bp, tilt):
    """Chance that a customer arrives at one quote in one minute."""
    decay = math.exp(-max(distance_bp, 0.0) / p.decay_bp)
    return p.base_rate * decay * (1 + tilt)


def customer_orders(p, w, m, bid, ask, fair):
    """Investors who trade with the market maker in minute m, as
    (change in its inventory, price)."""
    tilt = w.tilt[m]
    to_ask = (ask / fair - 1) / BP
    to_bid = (1 - bid / fair) / BP
    orders = []
    if w.u[0][m] < chance(p, to_ask, tilt):  # a buyer takes the offer
        orders.append((-w.sizes[0][m], ask))
    if w.u[1][m] < chance(p, to_bid, -tilt):  # a seller hits the bid
        orders.append((w.sizes[1][m], bid))
    return orders


def arbitrage_orders(p, w, m, bid, ask, fair, basket_bp):
    """An outside arbitrageur who trades only when a quote is far
    enough from fair value to pay for a creation or redemption."""
    if not p.primary or w.u[2][m] >= p.arb_rate:
        return []
    cost = creation_cost_bp(p, fair, basket_bp) * BP
    if bid > fair * (1 + cost):  # sell to the market maker, create
        return [(p.arb_size, bid)]
    if ask < fair * (1 - cost):  # buy from it, redeem
        return [(-p.arb_size, ask)]
    return []


def creation_cost_bp(p, fair, basket_bp):
    """Cost of creating or redeeming, per share in bp: the fund's fee
    plus the half-spread paid on the basket."""
    return p.fee / (p.unit * fair) / BP + basket_bp


def hedge_ratio(p):
    """Minimum-variance futures hedge per dollar of the fund."""
    weight, beta, _ = basket(p)
    factor = p.factor_vol**2
    return weight @ beta * factor / (factor + p.future_noise**2)


def hedge(p, desk, fair, fut):
    """Trade the hedge when more than the band is unhedged.
    Returns the cost of the trade."""
    exposure = desk.inventory * fair

    if p.hedge == "futures":
        notional = fut * p.multiplier
        target = round(-exposure * desk.ratio / notional)
        open_risk = exposure * desk.ratio + desk.contracts * notional
        if abs(open_risk) <= p.hedge_band:
            return 0.0
        traded = abs(target - desk.contracts) * notional
        desk.contracts = target
        cost = traded * p.future_cost_bp * BP

    elif p.hedge == "basket":
        gap = desk.inventory + desk.basket
        if abs(gap) * fair <= p.hedge_band:
            return 0.0
        desk.cash += gap * fair  # sell the excess basket or buy more
        desk.basket -= gap
        cost = abs(gap) * fair * desk.basket_bp * BP
    else:
        return 0.0

    desk.cash -= cost
    return cost


def create_or_redeem(p, desk, fair):
    """At the close, create whole units if short at least the
    threshold, or redeem if long. Returns the cost and the shares."""
    units = int(abs(desk.inventory) // p.unit)
    if not p.primary or units < p.threshold:
        return 0.0, 0
    sign = 1 if desk.inventory < 0 else -1  # +1 creates new shares
    shares = sign * units * p.unit

    # a basket hedge is the basket the fund wants, so it goes first
    from_hedge = max(0.0, min(units * p.unit, sign * desk.basket))
    desk.basket -= sign * from_hedge
    traded = units * p.unit - from_hedge
    desk.cash -= sign * traded * fair  # buy or sell the rest
    desk.inventory += shares

    cost = units * p.fee + traded * fair * desk.basket_bp * BP
    desk.cash -= cost
    return cost, abs(shares)


def minute(p, w, desk, m, pnl):
    """Quote, trade, hedge and mark the book in minute m, adding each
    piece of P&L to the day's totals. Returns the quotes and the
    shares traded."""
    fair, day = w.nav[m], m // MINUTES
    bid, ask = quotes(p, fair, desk.inventory)

    # fill every order at the standing quote, then hedge
    orders = customer_orders(p, w, m, bid, ask, fair)
    orders += arbitrage_orders(p, w, m, bid, ask, fair, desk.basket_bp)
    for shares, price in orders:
        desk.cash -= shares * price
        desk.inventory += shares
        pnl["edge"][day] += shares * (fair - price)
    pnl["hedging"][day] -= hedge(p, desk, fair, w.fut[m])

    # the market moves, and the futures settle their gain in cash
    move = w.nav[m + 1] - fair
    fut_move = (w.fut[m + 1] - w.fut[m]) * p.multiplier
    desk.cash += desk.contracts * fut_move
    pnl["inventory"][day] += desk.inventory * move
    pnl["hedge"][day] += desk.basket * move + desk.contracts * fut_move
    return bid, ask, sum(abs(s) for s, _ in orders)


MINUTES, YEAR_DAYS = 390, 252
DT = 1 / (YEAR_DAYS * MINUTES)  # one minute, in trading years
PIECES = ("edge", "inventory", "hedge", "hedging", "primary")
HEDGES = ("none", "futures", "basket")


@dataclass(frozen=True)
class Market:
    """The basket, the fund and the future."""

    # the basket and the fund
    stocks: int = 50
    factor_vol: float = 0.18  # the market factor, a year
    idio_vol: float = 0.30  # each stock's own news, a year
    beta_low: float = 0.8  # betas run evenly from low to high
    beta_high: float = 1.2
    weight_decay: float = 0.8  # weight in proportion to rank ** -0.8
    spread_low: float = 2.0  # half-spread of the largest holding, bp
    spread_high: float = 6.0  # half-spread of the smallest, bp
    nav0: float = 100.0  # value per share at the start
    unit: int = 50_000  # shares in a creation unit

    # the future
    fut0: float = 5_000.0  # index level at the start
    multiplier: int = 50  # dollars per index point
    future_noise: float = 0.01  # its index's own moves, a year
    future_cost_bp: float = 0.3  # cost of trading it


@dataclass(frozen=True)
class Params(Market):
    """Every setting of one run."""

    # the customers
    base_rate: float = 0.5  # chance per minute and side at fair value
    decay_bp: float = 15.0  # distance that cuts arrivals by 1 / e
    min_size: int = 100  # shares
    max_size: int = 2_000
    tilt_vol: float = 0.3  # typical size of the demand tilt
    tilt_half_life: float = 2.0  # trading days
    arb_rate: float = 0.1  # chance per minute an arbitrageur looks
    arb_size: int = 10_000  # shares

    # the market maker
    tick: float = 0.01
    half_spread_bp: float = 1.0
    lean_bp: float = 1.5  # lean per creation unit held
    lean_cap: float = 10.0  # units beyond which the lean stops
    hedge: str = "futures"  # "none", "futures" or "basket"
    hedge_band: float = 300_000.0  # dollars left unhedged
    primary: bool = True  # creation and redemption allowed
    threshold: int = 1  # units held before creating or redeeming
    fee: float = 500.0  # per unit created or redeemed

    # the run
    days: int = 20
    seed: int = 0


def market(p, rng):
    """Fund value per share and futures price at every minute."""
    n = p.days * MINUTES
    root = math.sqrt(DT)
    factor = p.factor_vol * root * rng.standard_normal(n)
    idio = p.idio_vol * root * rng.standard_normal((n, p.stocks))
    noise = p.future_noise * root * rng.standard_normal(n)

    nav = fair_value(p, factor, idio)
    var = p.factor_vol**2 + p.future_noise**2
    steps = factor + noise - 0.5 * var * DT
    fut = p.fut0 * np.exp(np.r_[0.0, np.cumsum(steps)])
    return nav, fut


def demand_tilt(p, rng, n):
    """A slowly mean-reverting tilt of demand towards buying
    (positive) or selling (negative)."""
    keep = 0.5 ** (1 / (p.tilt_half_life * MINUTES))
    scale = p.tilt_vol * math.sqrt(1 - keep**2)
    shocks = scale * rng.standard_normal(n)
    start = p.tilt_vol * rng.standard_normal()
    tilt, _ = lfilter([1.0], [1.0, -keep], shocks, zi=[keep * start])
    return np.clip(tilt, -0.9, 0.9)


@dataclass(frozen=True)
class World:
    """Every random input of one month, drawn from one seed."""

    nav: list  # fund value per share at each minute
    fut: list  # futures price at each minute
    tilt: list  # demand tilt at each minute
    u: list  # uniform draws for buyers, sellers and arbitrageurs
    sizes: list  # buyers' and sellers' order sizes


def draw(p):
    """Draw the market and the customers of one month from p.seed."""
    seeds = np.random.SeedSequence(p.seed).spawn(2)
    rng_market, rng_flow = (np.random.default_rng(s) for s in seeds)
    nav, fut = market(p, rng_market)

    n = p.days * MINUTES
    tilt = demand_tilt(p, rng_flow, n)
    u = rng_flow.random((3, n))
    sizes = rng_flow.integers(p.min_size, p.max_size + 1, (2, n))
    lists = (nav, fut, tilt, u, sizes)
    return World(*(x.tolist() for x in lists))


@dataclass
class Desk:
    """What the market maker holds."""

    ratio: float  # futures hedge per dollar of the fund
    basket_bp: float  # the basket's average half-spread
    inventory: float = 0.0  # fund shares, long positive
    basket: float = 0.0  # basket hedge, in fund shares
    contracts: int = 0  # futures hedge
    cash: float = 0.0


def simulate(p):
    """One simulated month, run one minute at a time."""
    w = draw(p)
    weight, _, half_spread = basket(p)
    desk = Desk(hedge_ratio(p), float(weight @ half_spread))
    pnl = {x: [0.0] * p.days for x in PIECES}
    n = p.days * MINUTES
    premium, inventory = np.zeros(n), np.zeros(n + 1)
    traded, events = 0.0, []

    for m in range(n):
        bid, ask, shares = minute(p, w, desk, m, pnl)
        premium[m] = ((bid + ask) / 2 / w.nav[m] - 1) / BP
        traded += shares
        if (m + 1) % MINUTES == 0:
            day, fair = m // MINUTES, w.nav[m + 1]
            cost, created = create_or_redeem(p, desk, fair)
            pnl["primary"][day] -= cost
            pnl["hedging"][day] -= hedge(p, desk, fair, w.fut[m + 1])
            events.append(created)
        inventory[m + 1] = desk.inventory

    equity = desk.cash + (desk.inventory + desk.basket) * w.nav[-1]
    days = {x: np.array(v) for x, v in pnl.items()}
    days["tracking"] = days["inventory"] + days["hedge"]
    days["total"] = sum(days[x] for x in PIECES)
    run = dict(premium=premium, inventory=inventory, traded=traded)
    return run | dict(days=days, events=events, equity=equity)


SEEDS = 40
BASE = Params()
_settings = {}
COLUMNS = (*PIECES, "tracking", "total")


def setting(p):
    """Averages over SEEDS months, with standard errors from the
    monthly averages. Each setting runs once and is kept."""
    if p in _settings:
        return _settings[p]
    months = [simulate(replace(p, seed=s)) for s in range(SEEDS)]
    per_month = {}
    for x in COLUMNS:
        per_month[x] = np.array([mo["days"][x].mean() for mo in months])

    out = {x: float(v.mean()) for x, v in per_month.items()}
    for x, v in per_month.items():
        out[f"{x}_se"] = float(v.std(ddof=1) / math.sqrt(SEEDS))
    out["per_month"] = per_month
    _settings[p] = out | describe(p, months)
    return _settings[p]


def paired(a, b, x="total"):
    """Mean and standard error of b minus a in one column of daily
    P&L, compared month by month on the same random markets."""
    d = b["per_month"][x] - a["per_month"][x]
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def band_bp(p):
    """Cost of creating per share at the starting value, in bp."""
    weight, _, half_spread = basket(p)
    return creation_cost_bp(p, p.nav0, float(weight @ half_spread))


def describe(p, months):
    """What the article reports about a setting beyond its means."""
    pool = {}
    for x in ("total", "tracking"):
        pool[x] = np.concatenate([mo["days"][x] for mo in months])
    signed = np.concatenate([mo["premium"] for mo in months])
    premium = np.abs(signed)
    units = [np.abs(mo["inventory"]).max() / p.unit for mo in months]
    traded = sum(mo["traded"] for mo in months)
    created = sum(sum(mo["events"]) for mo in months)
    events = sum(sum(e > 0 for e in mo["events"]) for mo in months)

    return {
        "daily_std": float(pool["total"].std(ddof=1)),
        "tracking_var": float(pool["tracking"].var(ddof=1)),
        "premium_std": float(signed.std()),
        "premium_p99": float(np.quantile(premium, 0.99)),
        "inside_band_pct": float(100 * (premium <= band_bp(p)).mean()),
        "max_units": float(max(units)),
        "traded_per_day": traded / (SEEDS * p.days),
        "netted_pct": 100 * (1 - created / traded),
        "events_per_month": events / SEEDS,
        "band_bp": band_bp(p),
    }


THRESHOLDS = (1, 2, 3, 5)
FEES = (0.0, 500.0, 2_000.0, 5_000.0, 10_000.0)


def sweeps():
    """Run every setting the article compares."""
    r = {"base": setting(BASE)}
    r["hedges"] = [setting(replace(BASE, hedge=h)) for h in HEDGES]
    r["off"] = setting(replace(BASE, primary=False))
    r["thresholds"] = [
        setting(replace(BASE, threshold=t)) for t in THRESHOLDS
    ]
    r["fees"] = [setting(replace(BASE, fee=f)) for f in FEES]
    return r


def fitted_ratio(p, seeds=5):
    """Slope of the fund's minute returns on the future's."""
    x, y = [], []
    for s in range(seeds):
        # the same market draws as month s
        seed = np.random.SeedSequence(s).spawn(2)[0]
        nav, fut = market(p, np.random.default_rng(seed))
        y.append(np.diff(nav) / nav[:-1])
        x.append(np.diff(fut) / fut[:-1])
    x, y = np.concatenate(x), np.concatenate(y)
    return float(np.cov(x, y)[0, 1] / x.var(ddof=1))


def tracking_error(p):
    """Yearly volatility of the fund less its futures hedge."""
    weight, beta, _ = basket(p)
    fund = (weight @ beta) ** 2 * p.factor_vol**2
    fund += (weight**2).sum() * p.idio_vol**2
    hedge = hedge_ratio(p) ** 2 * (p.factor_vol**2 + p.future_noise**2)
    return math.sqrt(fund - hedge)


SHOWN = ("stocks", "nav0", "half_spread_bp", "lean_bp", "unit", "fee")
SHOWN += ("lean_cap", "tilt_half_life", "future_cost_bp")


def model_numbers(r):
    """Numbers for the sections that build the model."""
    weight, _, half_spread = basket(BASE)
    out = {x: getattr(BASE, x) for x in SHOWN}
    out["months"] = SEEDS
    out["top_weight_pct"] = 100 * weight[0]
    out["bottom_weight_pct"] = 100 * weight[-1]
    out["unit_value"] = BASE.unit * BASE.nav0
    out["tilt_vol_pct"] = 100 * BASE.tilt_vol

    out["basket_bp"] = float(weight @ half_spread)
    out["hedge_ratio"] = float(hedge_ratio(BASE))
    out["hedge_ratio_fit"] = fitted_ratio(BASE)
    out["tracking_error_pct"] = 100 * tracking_error(BASE)
    out["band_bp"] = band_bp(BASE)

    b = r["base"]
    out["kept_pct"] = 100 * b["total"] / b["edge"]
    out["traded_per_day"] = b["traded_per_day"]
    out["netted_pct"] = b["netted_pct"]
    return out


def hedge_numbers(r):
    """Numbers for the section on hedges."""
    none, fut, bask = r["hedges"]
    out = {}
    for name, s in zip(HEDGES, r["hedges"], strict=True):
        out[f"{name}_cost"] = -(s["hedging"] + s["primary"])
        out[f"{name}_std"] = s["daily_std"]
        cut = s["tracking_var"] / none["tracking_var"]
        out[f"{name}_effectiveness_pct"] = 100 * (1 - cut)

    out["std_ratio"] = fut["daily_std"] / bask["daily_std"]
    gain, se = paired(bask, fut)
    out["futures_gain"], out["futures_gain_se"] = gain, se
    gain, se = paired(none, fut)
    out["futures_vs_none"], out["futures_vs_none_se"] = gain, se
    out["basket_tracking"] = bask["tracking"]
    out["basket_tracking_se"] = bask["tracking_se"]

    # basket the basket hedge traded, per fund share traded
    basket_bp = basket(BASE)[0] @ basket(BASE)[2]
    bought = -bask["hedging"] / (basket_bp * BP * BASE.nav0)
    out["basket_traded_pct"] = 100 * bought / bask["traded_per_day"]
    out["created_pct"] = 100 - fut["netted_pct"]
    return out


def primary_numbers(r):
    """Numbers for creation on and off, and the threshold and fees."""
    on, off = r["base"], r["off"]
    out = {}
    for name, s in (("on", on), ("off", off)):
        out[f"{name}_premium_std"] = s["premium_std"]
        out[f"{name}_premium_p99"] = s["premium_p99"]
        out[f"{name}_inside_pct"] = s["inside_band_pct"]
        out[f"{name}_max_units"] = s["max_units"]
        out[f"{name}_std"] = s["daily_std"]

    out["off_gain"], out["off_gain_se"] = paired(on, off)
    ratio = off["premium_std"] / on["premium_std"]
    out["premium_std_ratio"] = ratio

    for t, s in zip(THRESHOLDS, r["thresholds"], strict=True):
        out[f"t{t}_gain"], out[f"t{t}_gain_se"] = paired(on, s)
        out[f"t{t}_events"] = s["events_per_month"]
        out[f"t{t}_primary"] = -s["primary"]
        out[f"t{t}_std"] = s["daily_std"]
        out[f"t{t}_max_units"] = s["max_units"]

    out["fee_flat_from"] = FEES[2]
    for f, s in zip(FEES, r["fees"], strict=True):
        out[f"fee{int(f)}_band_bp"] = s["band_bp"]
        out[f"fee{int(f)}_p99"] = s["premium_p99"]
    return out


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def money(v):
    return f"{v:,.0f}"


def example_table(days):
    """The example month in dollars, summed by week."""
    rows = []
    for week in range(4):
        part = slice(5 * week, 5 * week + 5)
        cells = [money(days[x][part].sum()) for x in COLUMNS]
        rows.append([week + 1, *cells])
    rows.append(["all", *[money(days[x].sum()) for x in COLUMNS]])
    header = ["week", "edge", "inventory", "hedge", "hedging"]
    header += ["primary", "tracking", "total"]
    return table(rows, header, [4, 8, 9, 8, 7, 7, 8, 8])


def hedge_table(r):
    """Daily cost, swing and P&L of each hedge, in dollars."""
    rows = []
    for name, s in zip(HEDGES, r["hedges"], strict=True):
        cells = [s["hedging"], s["primary"], s["daily_std"], s["total"]]
        rows.append([name, *[money(c) for c in cells]])
    header = ["hedge", "hedging", "primary", "daily std", "total"]
    return table(rows, header, [7, 7, 7, 9, 6])


def threshold_table(r):
    """Creations, cost, swing, P&L and premium by threshold."""
    rows = []
    for t, s in zip(THRESHOLDS, r["thresholds"], strict=True):
        gain, se = paired(r["base"], s)
        cells = [f"{s['events_per_month']:.1f}", money(s["primary"])]
        cells += [money(s["daily_std"]), money(s["total"])]
        cells += [money(gain), money(se), f"{s['premium_p99']:.1f}"]
        rows.append([t, *cells])
    header = ["units", "per month", "primary", "daily std", "total"]
    header += ["vs 1 unit", "std error", "premium p99"]
    return table(rows, header, [5, 9, 7, 9, 5, 9, 9, 11])


EXAMPLE_SEED = 3
NUMBERS = (model_numbers, hedge_numbers, primary_numbers)


def results():
    """Every number and table the article quotes, ready for JSON."""
    r = sweeps()
    derived = {}
    for numbers in NUMBERS:
        derived |= numbers(r)

    example = simulate(replace(BASE, seed=EXAMPLE_SEED))["days"]
    tables = {
        "example": example_table(example),
        "hedges": hedge_table(r),
        "thresholds": threshold_table(r),
    }
    out = {"base": clean(r["base"]), "derived": derived}
    out["tables"] = tables
    for name in ("hedges", "thresholds", "fees"):
        out[name] = [clean(s) for s in r[name]]
    out["off"] = clean(r["off"])
    return out


def clean(s):
    """A setting without its per-month arrays."""
    return {x: v for x, v in s.items() if x != "per_month"}


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


def chart_hedges(res, out):
    """Cost per day and the swing of daily P&L for each hedge."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    where = range(len(HEDGES))
    cost = [-(s["hedging"] + s["primary"]) for s in res["hedges"]]
    swing = [s["daily_std"] for s in res["hedges"]]
    style = dict(width=0.5, edgecolor=SURFACE, lw=0.8)
    ax[0].bar(where, cost, color=BLUE, **style)
    ax[1].bar(where, swing, color=ORANGE, **style)
    ax[1].set_xticks(where, ["no hedge", "futures", "basket"])

    finish(ax[0], "Hedging and creating, cost per day", money=True)
    finish(ax[1], "Standard deviation of daily P&L", money=True)
    fig.savefig(out / "hedges.png", metadata={"Software": None})
    plt.close(fig)


def premium_panel(ax, primary, colour):
    """One run of the example month's premium, between the band's
    edges. Returns the premium at every minute."""
    p = replace(BASE, seed=EXAMPLE_SEED, primary=primary)
    premium = simulate(p)["premium"]
    days = np.arange(len(premium)) / MINUTES
    ax.plot(days, premium, color=colour, lw=0.8)

    band = band_bp(BASE)
    for edge in (band, -band):
        ax.plot([0, BASE.days], [edge, edge], c=GREY, lw=0.8, ls="--")
    return premium


def chart_premium(res, out):
    """One month's premium with creation and redemption on and off."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    on = premium_panel(ax[0], True, BLUE)
    off = premium_panel(ax[1], False, ORANGE)
    ax[1].set_xticks(range(0, BASE.days + 1, 5))
    ax[1].set_xlabel("trading days into the month")

    finish(ax[0], "Premium in bp, creation and redemption allowed")
    finish(ax[1], "Premium in bp, creation and redemption switched off")
    fig.savefig(out / "premium.png", metadata={"Software": None})
    plt.close(fig)

    sampled = slice(None, None, MINUTES // 6)
    band = [band_bp(BASE)] * len(on[sampled])
    res["cover"] = {"on": on[sampled].tolist()}
    res["cover"]["off"] = off[sampled].tolist()
    res["cover"]["band_high"] = band
    res["cover"]["band_low"] = [-b for b in band]


def chart_fees(res, out):
    """The 99th percentile premium against the cost of creating."""
    fig, ax = plt.subplots()
    cost = [s["band_bp"] for s in res["fees"]]
    p99 = [s["premium_p99"] for s in res["fees"]]
    top = MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(0, max(cost))
    edge = "cost of creating"
    ax.plot([0, top[-1]], [0, top[-1]], c=GREY, lw=0.8, label=edge)

    label = "99th percentile of the premium"
    ax.plot(cost, p99, color=BLUE, marker="o", label=label)
    ax.set_xticks(top)
    ax.set_xlim(0, top[-1])
    ax.set_xlabel("cost of creating or redeeming per share, in bp")

    finish(
        ax, "Premium against the cost of creating, in bp", legend=True
    )
    fig.savefig(out / "fees.png", metadata={"Software": None})
    plt.close(fig)


def main(out=Path(".")):
    res = results()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chart_hedges(res, figures)
    chart_premium(res, figures)
    chart_fees(res, figures)

    text = json.dumps(res, indent=1, default=float) + "\n"
    (out / "results.json").write_text(text)
    print(res["tables"]["example"], res["tables"]["hedges"], sep="\n\n")
    print(res["tables"]["thresholds"])


if __name__ == "__main__":
    main()
