# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr


def greeks(s, k, tau, vol, is_call):
    """Black-Scholes price and Greeks with zero rates and dividends.

    Vega is per 1.00 of vol and theta is per year.
    """
    sd = vol * np.sqrt(tau)
    d1 = (np.log(s / k) + 0.5 * sd**2) / sd
    pdf = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    call = s * ndtr(d1) - k * ndtr(d1 - sd)

    # the put's price follows from put-call parity
    return {
        "price": np.where(is_call, call, call - s + k),
        "delta": np.where(is_call, ndtr(d1), ndtr(d1) - 1),
        "gamma": pdf / (s * sd),
        "vega": s * pdf * np.sqrt(tau),
        "theta": -s * pdf * vol / (2 * np.sqrt(tau)),
    }


def quotes(p, theo, vega, net_vega):
    """Bid and offer for each option from its theo and vega per share
    and the book's net vega in dollars per vol point."""
    vega_pt = 0.01 * vega  # premium per share per vol point

    # lean against the position, and keep at least a tick of width
    lean = -p.lean_vol * np.clip(net_vega / p.vega_limit, -2, 2)
    mid = theo + lean * vega_pt
    half_width = np.maximum(p.tick, p.half_width_vol * vega_pt)

    # round away from the leaned mid onto the tick grid
    bid = np.floor((mid - half_width) / p.tick + 1e-9) * p.tick
    ask = np.ceil((mid + half_width) / p.tick - 1e-9) * p.tick
    bid = np.where(bid >= p.tick, bid, np.nan)  # no bid below a tick

    # at the vega limit, stop quoting the side that adds risk
    if net_vega >= p.vega_limit:
        bid[:] = np.nan
    if net_vega <= -p.vega_limit:
        ask[:] = np.nan
    return bid, ask


def arrival_rate(p, distance):
    """Uninformed customers per minute on one side of one quote.

    The rate falls exponentially with the quote's distance from theo,
    measured in vol points.
    """
    decay = np.exp(-np.maximum(distance, 0) / p.width_sensitivity)
    return 0.5 * p.base_rate * decay


def uninformed_orders(p, w, m, bid, ask, theo, vega):
    """Random buyers and sellers who reach each quote in minute m, as
    (option, +1 if the customer buys or -1 if it sells, kind, size)."""
    vega_pt = 0.01 * vega
    with np.errstate(invalid="ignore"):  # a missing quote has no takers
        rate_buy = arrival_rate(p, (ask - theo) / vega_pt)
        rate_sell = arrival_rate(p, (theo - bid) / vega_pt)

    buyers = np.flatnonzero(w.u[0, m] < np.nan_to_num(rate_buy))
    sellers = np.flatnonzero(w.u[1, m] < np.nan_to_num(rate_sell))
    orders = [(i, +1, "uninformed", w.sizes[0, m, i]) for i in buyers]
    orders += [(i, -1, "uninformed", w.sizes[0, m, i]) for i in sellers]
    return orders


def theo_ahead(w, m, i, horizon):
    """Theo of option i at minute m, priced with the vol level
    `horizon` minutes later, which vol-informed traders know."""
    future = w.level[min(m + horizon, len(w.level) - 1)]
    vol = max(future + w.smile[i], 0.03)
    g = greeks(w.spot[m], w.k[i], w.tau[m], vol, w.is_call[i])
    return float(g["price"])


def informed_orders(p, w, m, bid, ask, theo_next):
    """Orders from fast and vol-informed traders in minute m. They
    arrive at random and trade only when a quote is wrong."""
    orders = []
    fast = np.flatnonzero(w.u[2, m] < p.base_rate * p.fast_ratio)
    for i in fast:  # they see the next minute's theo
        if theo_next[i] > ask[i]:
            orders.append((i, +1, "fast", w.sizes[1, m, i]))
        elif theo_next[i] < bid[i]:
            orders.append((i, -1, "fast", w.sizes[1, m, i]))

    chance = p.base_rate * p.vol_informed_ratio
    for i in np.flatnonzero(w.u[3, m] < chance):
        future = theo_ahead(w, m, i, p.vol_horizon)
        if future > ask[i]:
            orders.append((i, +1, "vol-informed", w.sizes[1, m, i]))
        elif future < bid[i]:
            orders.append((i, -1, "vol-informed", w.sizes[1, m, i]))
    return orders


def explain(book, m, g0, g1, ds, dv, mult):
    """Split the P&L of the book held through minute m into pieces,
    using the Greeks from the start of the minute."""
    pos, pnl = book.pos, book.pnl
    pnl["delta"][m] = (pos @ g0["delta"] * mult + book.shares) * ds
    pnl["gamma"][m] = 0.5 * (pos @ g0["gamma"]) * mult * ds**2
    pnl["theta"][m] = (pos @ g0["theta"]) * mult * DT
    pnl["vega"][m] = (pos @ g0["vega"]) * mult * dv

    actual = pos @ (g1["price"] - g0["price"]) * mult + book.shares * ds
    explained = sum(pnl[x][m] for x in GREEK_TERMS)
    pnl["other"][m] = actual - explained


def markouts(run, p):
    """Sum and count of hedged markouts per contract, by kind of
    customer and minutes after the fill."""
    f = fill_arrays(run)
    m, i, side = f["m"], f["i"], f["side"]
    spot, k, is_call = run["spot"], run["k"][i], run["is_call"][i]
    out = {kd: {} for kd in KINDS}
    for h in HORIZONS:
        # theo h minutes later, with time to expiry held at the fill's
        vol = np.maximum(run["level"][m + h] + run["smile"][i], 0.03)
        g = greeks(spot[m + h], k, run["tau"][m], vol, is_call)
        value = (g["price"] - f["price"]) * side * p.multiplier

        if h >= 1:  # the one-off hedge at the end of the fill's minute
            move = spot[m + h] - spot[m + 1]
            value -= f["delta"] * side * move * p.multiplier

        for kd in KINDS:
            sel = f["kind"] == kd
            out[kd][h] = (float(value[sel].sum()), int(sel.sum()))
    return out


MINUTES, YEAR_DAYS = 390, 252
DT = 1 / (YEAR_DAYS * MINUTES)  # one minute, in trading years

# strikes as a share of the opening stock price
STRIKES = (0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15)

GREEK_TERMS = ("delta", "gamma", "theta", "vega")
PIECES = ("edge", "new_fill", *GREEK_TERMS, "other", "hedging")


@dataclass
class Book:
    """What the market maker holds, and its P&L so far."""

    pos: np.ndarray  # contracts of each option, long positive
    pnl: dict  # each piece of P&L, minute by minute
    fills: list = field(default_factory=list)
    shares: float = 0.0  # the stock hedge
    cash: float = 0.0


@dataclass(frozen=True)
class Params:
    # the stock and its vol level
    s0: float = 100.0  # opening stock price
    realized_vol: float = 0.20  # how much the stock really moves
    vol_mean: float = 0.20  # the vol level reverts to this
    vol_reversion: float = 5.0  # speed of that reversion, per year
    vol_of_vol: float = 0.16  # about one vol point a day
    spot_vol_corr: float = -0.5  # vol tends to rise as the stock falls

    # the options
    skew: float = -0.10  # vol change per unit of ln(strike / s0)
    curvature: float = 0.50  # how much the smile bends
    strikes: tuple = STRIKES
    expiry_days: float = 45.0  # trading days
    multiplier: int = 100  # shares per contract

    # the customers
    base_rate: float = 0.04  # uninformed per option-minute at theo
    width_sensitivity: float = 1.0  # vol points
    fast_ratio: float = 0.25  # fast traders per uninformed customer
    vol_informed_ratio: float = 0.25  # vol-informed per uninformed
    vol_horizon: int = 120  # minutes the vol-informed see ahead
    max_size: int = 10  # contracts per order, uniform from 1

    # the quotes, the hedge and the run
    half_width_vol: float = 0.5  # vol points either side of theo
    tick: float = 0.01
    vega_limit: float = 10_000.0  # dollars per vol point
    lean_vol: float = 1.0  # vol points of lean at the vega limit
    hedge_band: float = 0.0  # shares of delta tolerated unhedged
    hedge_cost: float = 0.01  # dollars per share hedged
    days: int = 5  # sessions of 390 minutes
    seed: int = 0


def options(p):
    """Strike, call or put, and smile offset of each option."""
    k = p.s0 * np.repeat(p.strikes, 2)
    is_call = np.tile([True, False], len(p.strikes))
    x = np.log(k / p.s0)
    smile = p.skew * x + p.curvature * x**2
    return k, is_call, smile


def paths(p, rng):
    """The stock price and the vol level at every minute of the week."""
    n = p.days * MINUTES
    rho = p.spot_vol_corr
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.standard_normal(n)

    # the stock follows a geometric Brownian motion with no drift
    vol = p.realized_vol
    steps = -0.5 * vol**2 * DT + vol * math.sqrt(DT) * z1
    spot = p.s0 * np.exp(np.r_[0.0, np.cumsum(steps)])

    # the vol level is pulled back towards its mean every minute
    level = np.empty(n + 1)
    level[0] = p.vol_mean
    for i in range(n):
        pull = p.vol_reversion * (p.vol_mean - level[i]) * DT
        shock = p.vol_of_vol * math.sqrt(DT) * z2[i]
        level[i + 1] = level[i] + pull + shock
    return spot, level


@dataclass(frozen=True)
class World:
    """Every random input of one week, drawn from one seed."""

    k: np.ndarray  # strike of each option
    is_call: np.ndarray
    smile: np.ndarray  # each option's vol offset from the vol level
    spot: np.ndarray  # stock price at each minute
    level: np.ndarray  # vol level at each minute
    tau: np.ndarray  # years to expiry at each minute
    u: np.ndarray  # uniform draws that decide who arrives
    sizes: np.ndarray  # order sizes in contracts

    def greeks_at(self, m, level):
        """Greeks of every option at minute m for a given vol level."""
        vol = np.maximum(level + self.smile, 0.03)
        s, tau = self.spot[m], self.tau[m]
        return greeks(s, self.k, tau, vol, self.is_call)


def draw(p):
    """Draw the market and the customers of one week from p.seed."""
    seeds = np.random.SeedSequence(p.seed).spawn(2)
    rng_path, rng_flow = (np.random.default_rng(s) for s in seeds)
    k, is_call, smile = options(p)
    spot, level = paths(p, rng_path)

    n = p.days * MINUTES
    tau = p.expiry_days / YEAR_DAYS - DT * np.arange(n + 1)
    # arrivals of the four kinds of order, then the order sizes
    u = rng_flow.random((4, n, len(k)))
    sizes = rng_flow.integers(1, p.max_size + 1, (2, n, len(k)))
    return World(k, is_call, smile, spot, level, tau, u, sizes)


def trade(book, m, orders, bid, ask, g0, g1, mult):
    """Fill each order at the standing quote, if that side is quoted."""
    for i, side, kind, size in orders:
        price = ask[i] if side > 0 else bid[i]
        if np.isnan(price):
            continue

        q = -side * size  # the market maker takes the other side
        book.cash -= q * price * mult
        book.pos[i] += q
        book.pnl["edge"][m] += (g0["price"][i] - price) * q * mult
        move = g1["price"][i] - g0["price"][i]
        book.pnl["new_fill"][m] += move * q * mult
        fill = (m, i, q, price, kind, g0["delta"][i], g0["vega"][i])
        book.fills.append(fill)


def hedge(p, book, m, spot, option_delta):
    """Trade the stock back to flat if net delta is outside the band."""
    delta_now = option_delta + book.shares
    if abs(delta_now) > p.hedge_band:
        h = -round(delta_now)
        book.cash -= h * spot + p.hedge_cost * abs(h)
        book.shares += h
        book.pnl["hedging"][m] = -p.hedge_cost * abs(h)


def minute(p, w, book, m, g0, g1):
    """Quote, explain, fill and hedge in minute m. Returns net vega."""
    mult = p.multiplier
    theo, vega = g0["price"], g0["vega"]
    net_vega = book.pos @ vega * 0.01 * mult  # dollars per vol point
    bid, ask = quotes(p, theo, vega, net_vega)

    ds = w.spot[m + 1] - w.spot[m]
    dv = w.level[m + 1] - w.level[m]
    explain(book, m, g0, g1, ds, dv, mult)

    orders = informed_orders(p, w, m, bid, ask, g1["price"])
    orders += uninformed_orders(p, w, m, bid, ask, theo, vega)
    trade(book, m, orders, bid, ask, g0, g1, mult)

    hedge(p, book, m, w.spot[m + 1], book.pos @ g1["delta"] * mult)
    return net_vega


def simulate(p):
    """One simulated week, run one minute at a time."""
    w = draw(p)
    n = len(w.spot) - 1
    book = Book(np.zeros(len(w.k)), {x: np.zeros(n) for x in PIECES})
    equity, net_vega = np.zeros(n + 1), np.zeros(n)

    g0 = w.greeks_at(0, w.level[0])
    for m in range(n):
        g1 = w.greeks_at(m + 1, w.level[m + 1])
        net_vega[m] = minute(p, w, book, m, g0, g1)
        hedge_value = book.shares * w.spot[m + 1]
        options_value = book.pos @ g1["price"] * p.multiplier
        equity[m + 1] = book.cash + hedge_value + options_value
        g0 = g1

    run = dict(pnl=book.pnl, fills=book.fills, net_vega=net_vega)
    return dict(vars(w), equity=equity, **run)


SEEDS = 40
BASE = Params()
_settings = {}


def daily(run, p):
    """Each P&L piece summed over each day, and the day's total."""
    days = {}
    for x, v in run["pnl"].items():
        days[x] = v.reshape(p.days, MINUTES).sum(axis=1)
    days["total"] = sum(days[x] for x in PIECES)
    return days


def week(p):
    """One simulated week, reduced to what the experiments need."""
    run = simulate(p)
    vega = np.abs(run["net_vega"])
    return {
        "days": daily(run, p),
        "mean_abs_vega": float(vega.mean()),
        "max_abs_vega": float(vega.max()),
        "markouts": markouts(run, p),
        **fill_counts(run, p),
    }


def setting(p):
    """Averages over SEEDS weeks, with standard errors from the weekly
    averages. Each setting runs once and is kept."""
    if p in _settings:
        return _settings[p]
    weeks = [week(replace(p, seed=s)) for s in range(SEEDS)]
    per_week = {}
    for x in (*PIECES, "total"):
        per_week[x] = np.array([w["days"][x].mean() for w in weeks])

    out = {x: float(v.mean()) for x, v in per_week.items()}
    for x, v in per_week.items():
        out[f"{x}_se"] = float(v.std(ddof=1) / math.sqrt(SEEDS))
    out["per_week"] = per_week
    _settings[p] = out | describe(weeks, p)
    return _settings[p]


def paired(a, b, x="total"):
    """Mean and standard error of b minus a in one column of daily
    P&L, compared week by week on the same random markets."""
    d = b["per_week"][x] - a["per_week"][x]
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def informed(ratio, p=BASE):
    """A setting with `ratio` informed traders per uninformed customer,
    split evenly between fast and vol-informed traders."""
    half = ratio / 2
    return replace(p, fast_ratio=half, vol_informed_ratio=half)


HORIZONS = (0, 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240)
KINDS = ("uninformed", "fast", "vol-informed")


def fill_arrays(run):
    """The week's fills as arrays, keeping only those with the longest
    horizon still to run."""
    columns = zip(*run["fills"], strict=True)
    m, i, q, price, kind, delta, _ = (np.array(c) for c in columns)
    ok = m + max(HORIZONS) < len(run["spot"])

    return {
        "m": m[ok],
        "i": i[ok].astype(int),
        "side": np.sign(q[ok].astype(float)),
        "price": price[ok].astype(float),
        "kind": kind[ok],
        "delta": delta[ok].astype(float),
    }


TOXICITY = (0.0, 0.25, 0.5, 1.0, 2.0)
WIDTHS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BANDS = (0, 100, 250, 500, 1000, 2000, 5000)
LEANS = (0.0, 0.5, 1.0, 2.0)


def sweeps():
    """Run every setting the article compares."""
    calm = informed(0.0)
    r = {"base": setting(BASE)}
    r["toxicity"] = [setting(informed(x)) for x in TOXICITY]
    r["leans"] = [setting(replace(BASE, lean_vol=v)) for v in LEANS]

    for name, p in (("width", BASE), ("width_calm", calm)):
        r[name] = [
            setting(replace(p, half_width_vol=w)) for w in WIDTHS
        ]

    bands = [replace(BASE, hedge_band=float(b)) for b in BANDS]
    r["bands"] = [setting(x) for x in bands]
    r["band_calm"] = setting(replace(calm, hedge_band=2000.0))
    return r


def fill_counts(run, p):
    """Contracts by kind of customer, and the fast traders' fills on
    options with a delta above 0.7."""
    fills = run["fills"]
    contracts = {}
    for kd in KINDS:
        contracts[kd] = sum(abs(f[2]) for f in fills if f[4] == kd)
    fast_hi = [f for f in fills if f[4] == "fast" and abs(f[5]) > 0.7]

    # the half-width before rounding, and delta times a one-minute
    # stock move of one standard deviation, both in cents
    widths, moves = [], []
    for m, i, q, price, kind, delta, vega in fast_hi:
        widths.append(100 * max(p.tick, p.half_width_vol * 0.01 * vega))
        cents = 100 * abs(delta) * run["spot"][m]
        moves.append(cents * p.realized_vol * math.sqrt(DT))

    return {
        "contracts": contracts,
        "fast_hi_contracts": sum(abs(f[2]) for f in fast_hi),
        "fast_hi_half_width_cents": widths,
        "fast_hi_move_cents": moves,
    }


def describe(weeks, p):
    """What the article reports about a setting beyond its means."""
    return (
        day_numbers(weeks)
        | flow_numbers(weeks, p)
        | {"markout": pooled_markouts(weeks)}
        | latency_numbers(weeks)
    )


def day_numbers(weeks):
    """The spread of daily P&L and the number of losing days."""
    all_days = np.concatenate([w["days"]["total"] for w in weeks])
    return {
        "daily_std": float(all_days.std(ddof=1)),
        "losing_days": int((all_days < 0).sum()),
        "days": len(all_days),
    }


def flow_numbers(weeks, p):
    """Contracts a day by kind of customer, and the book's net vega."""
    out = {}
    for kd in KINDS:
        traded = sum(w["contracts"][kd] for w in weeks)
        out[f"contracts_{kd}"] = traded / (SEEDS * p.days)
    out["contracts"] = sum(out[f"contracts_{kd}"] for kd in KINDS)

    vega = [w["mean_abs_vega"] for w in weeks]
    out["mean_abs_vega"] = float(np.mean(vega))
    out["max_abs_vega"] = max(w["max_abs_vega"] for w in weeks)
    return out


def pooled_markouts(weeks):
    """Markout per contract by kind and horizon, over all the fills."""
    out = {}
    for kd in KINDS:
        out[kd] = {}
        for h in HORIZONS:
            total = sum(w["markouts"][kd][h][0] for w in weeks)
            count = sum(w["markouts"][kd][h][1] for w in weeks)
            out[kd][h] = total / max(1, count)

    return out


def latency_numbers(weeks):
    """The fast traders' high-delta fills, pooled over the weeks."""
    fast = sum(w["contracts"]["fast"] for w in weeks)
    high = sum(w["fast_hi_contracts"] for w in weeks)
    widths, moves = [], []
    for w in weeks:
        widths += w["fast_hi_half_width_cents"]
        moves += w["fast_hi_move_cents"]

    return {
        "fast_hi_share": high / max(1, fast),
        "fast_hi_half_width_cents": median(widths),
        "fast_hi_move_cents": median(moves),
    }


def median(values):
    """The median of a list, or None when the list is empty."""
    return float(np.median(values)) if values else None


def pct_change(new, old):
    """Percentage change from old to new."""
    return 100 * (new / old - 1)


def pct_cut(new, old):
    """Percentage by which new is below old."""
    return 100 * (1 - new / old)


def explain_numbers(r):
    """Numbers for the sections on quoting and the P&L explain."""
    b = r["base"]
    tau = BASE.expiry_days / YEAR_DAYS
    atm = greeks(BASE.s0, BASE.s0, tau, BASE.vol_mean, True)
    half_width = BASE.half_width_vol * 0.01 * atm["vega"]
    fast, vol = b["contracts_fast"], b["contracts_vol-informed"]
    share = 100 * (fast + vol) / b["contracts"]

    per_contract = float(half_width * BASE.multiplier)

    out = {"vega_limit": BASE.vega_limit}
    out["atm_half_width_per_contract"] = per_contract
    out["informed_contract_share_pct"] = share
    out["kept_of_edge_pct"] = 100 * b["total"] / b["edge"]
    out["gamma_plus_theta"] = b["gamma"] + b["theta"]

    out["mean_over_std"] = b["total"] / b["daily_std"]
    out["total_days"] = b["days"]
    out["losing_days"] = b["losing_days"]
    return out


def toxicity_numbers(r):
    """Numbers for the section on informed traders."""
    none, even = r["toxicity"][0], r["toxicity"][3]
    out = {"tox0_vega": none["vega"], "tox0_vega_se": none["vega_se"]}
    contracts = pct_change(even["contracts"], none["contracts"])
    edge = pct_change(even["edge"], none["edge"])
    drop = pct_cut(even["total"], none["total"])

    out["tox_contracts_change_pct_0_to_100"] = contracts
    out["tox_edge_change_pct_0_to_100"] = edge
    out["tox_total_drop_pct_0_to_100"] = drop
    return out


def markout_numbers(r):
    """Numbers for the section on markouts and latency."""
    b = r["base"]
    mk = b["markout"]
    out = {"mk_uninformed_0": mk["uninformed"][0]}
    out["mk_uninformed_240"] = mk["uninformed"][240]
    out["mk_fast_0"] = mk["fast"][0]
    out["mk_fast_1_loss"] = -mk["fast"][1]
    out["mk_vol_0"] = mk["vol-informed"][0]
    out["mk_vol_120_loss"] = -mk["vol-informed"][120]

    out["fast_high_delta_pct"] = 100 * b["fast_hi_share"]
    out["fast_median_half_width_cents"] = b["fast_hi_half_width_cents"]
    out["fast_theo_move_1m_cents"] = b["fast_hi_move_cents"]
    return out


def best_width(r):
    """Index of the half-width with the highest daily P&L."""
    totals = [w["total"] for w in r["width"]]
    return totals.index(max(totals))


def width_numbers(r):
    """Numbers for the section on quote width."""
    wide, calm, best = r["width"], r["width_calm"], best_width(r)
    out = {"base_half_width": BASE.half_width_vol}
    out["best_width"] = WIDTHS[best]
    out["width_narrower"] = WIDTHS[best - 1]
    out["width_wider"] = WIDTHS[best + 1]
    out["best_width_total"] = wide[best]["total"]

    # the totals half a point either side of the best half-width
    below = WIDTHS.index(WIDTHS[best] - 0.5)
    above = WIDTHS.index(WIDTHS[best] + 0.5)
    out["half_below_best_total"] = wide[below]["total"]
    out["half_above_best_total"] = wide[above]["total"]

    out["narrowest_width"] = WIDTHS[0]
    out["narrowest_total"] = wide[0]["total"]
    out["narrowest_total_no_informed"] = calm[0]["total"]
    out["narrowest_contracts"] = wide[0]["contracts"]
    out["widest_width"] = WIDTHS[-1]
    out["widest_total"] = wide[-1]["total"]
    out["widest_total_no_informed"] = calm[-1]["total"]
    return out


def band_numbers(r):
    """Numbers for the section on hedge bands."""
    flat, b1000, b2000, b5000 = (r["bands"][j] for j in (0, 4, 5, 6))
    cost, std = flat["hedging"], flat["daily_std"]
    out = {"hedge0_cost": cost}
    out["hedge1000_cost_cut_pct"] = pct_cut(b1000["hedging"], cost)
    out["hedge1000_std_rise_pct"] = pct_change(b1000["daily_std"], std)
    out["hedge5000_std_rise_pct"] = pct_change(b5000["daily_std"], std)
    out["hedge2000_cost_saved"] = b2000["hedging"] - cost

    out["band2000_delta"] = b2000["delta"]
    out["band2000_delta_se"] = b2000["delta_se"]
    out["band2000_delta_no_informed"] = r["band_calm"]["delta"]
    out["band2000_delta_no_informed_se"] = r["band_calm"]["delta_se"]
    return out


def lean_numbers(r):
    """Numbers for the section on leaning the quotes."""
    none, half, one, two = r["leans"]
    vega, std = none["mean_abs_vega"], none["daily_std"]
    out = {"lean1_vega_cut_pct": pct_cut(one["mean_abs_vega"], vega)}
    out["lean1_std_cut_pct"] = pct_cut(one["daily_std"], std)
    out["lean2_std_cut_pct"] = pct_cut(two["daily_std"], std)
    out["lean0_max_vega"] = none["max_abs_vega"]
    out["lean1_max_vega"] = one["max_abs_vega"]
    return out


def paired_numbers(r):
    """Paired differences in daily P&L, each with its standard error."""
    tox, wide, calm = r["toxicity"], r["width"], r["width_calm"]
    bands, leans, best = r["bands"], r["leans"], best_width(r)
    below = WIDTHS.index(WIDTHS[best] - 0.5)
    above = WIDTHS.index(WIDTHS[best] + 0.5)

    pairs = {
        "tox_total_drop_0_to_100": (tox[3], tox[0]),
        "informed_cost_at_best_width": (wide[best], calm[best]),
        "best_width_beats_narrower": (wide[best - 1], wide[best]),
        "best_width_beats_wider": (wide[best + 1], wide[best]),
        "half_above_beats_half_below": (wide[below], wide[above]),
        "band2000_gain": (bands[0], bands[5]),
        "band5000_gain": (bands[0], bands[6]),
        "lean05_cost": (leans[1], leans[0]),
        "lean1_cost": (leans[2], leans[0]),
        "lean2_cost": (leans[3], leans[0]),
    }

    out = {}
    for name, (a, b) in pairs.items():
        out[name], out[f"{name}_se"] = paired(a, b)
    vega_cost = paired(r["base"], tox[0], "vega")
    out["informed_vega_cost"], out["informed_vega_cost_se"] = vega_cost
    return out


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def money(v):
    return f"{v:,.0f}"


SHOWN = ("edge", "new_fill", "gamma", "theta", "vega", "hedging")


def example_row(label, day):
    """One row of the example week, with delta folded into other."""
    cells = [money(day[x]) for x in SHOWN]
    other = money(day["delta"] + day["other"])
    return [label, *cells, other, money(day["total"])]


def example_table(ex):
    """The example week in dollars a day, and its sum."""
    rows = []
    for day in range(BASE.days):
        one_day = {x: v[day] for x, v in ex.items()}
        rows.append(example_row(day + 1, one_day))
    rows.append(example_row("sum", {x: v.sum() for x, v in ex.items()}))
    header = ["day", "edge", "new fill", "gamma", "theta", "vega"]
    header += ["hedging", "other", "total"]
    return table(rows, header, [3, 7, 8, 6, 6, 7, 7, 5, 7])


def toxicity_table(tox):
    """Daily averages for each number of informed traders."""
    rows = []
    for ratio, t in zip(TOXICITY, tox, strict=True):
        cells = [
            money(t[x]) for x in ("edge", "new_fill", "vega", "total")
        ]
        rows.append([int(100 * ratio), money(t["contracts"]), *cells])
    header = ["informed per 100", "contracts/day", "edge", "new fill"]
    header += ["vega", "total"]
    return table(rows, header, [16, 13, 6, 8, 6, 6])


def width_table(r):
    """Daily P&L at each half-width, and contracts traded."""
    rows = []
    pairs = zip(r["width_calm"], r["width"], strict=True)
    for width, (calm, base) in zip(WIDTHS, pairs, strict=True):
        totals = [money(calm["total"]), money(base["total"])]
        rows.append([f"{width:.2f}", *totals, money(base["contracts"])])

    header = ["half-width", "uninformed only", "with informed"]
    header += ["contracts/day"]
    return table(rows, header, [10, 15, 13, 13])


def lean_table(leans):
    """Cost, swing and net vega of each lean against no lean."""
    none = leans[0]
    rows = []
    for lean, s in zip(LEANS, leans, strict=True):
        cost, se = paired(s, none)
        cut = pct_cut(s["daily_std"], none["daily_std"])
        vega = [money(s["mean_abs_vega"]), money(s["max_abs_vega"])]
        cells = [money(cost), money(se), f"{cut:.0f}%"]
        rows.append([f"{lean:.1f}", *cells, *vega])

    header = ["lean", "cost", "std error", "swing cut"]
    header += ["average vega", "largest vega"]
    return table(rows, header, [4, 4, 9, 9, 12, 12])


NUMBERS = (explain_numbers, toxicity_numbers, markout_numbers)
NUMBERS += (width_numbers, band_numbers, lean_numbers, paired_numbers)


def results():
    """Every number and table the article quotes, ready for JSON."""
    r = sweeps()
    derived = {}
    for numbers in NUMBERS:
        derived |= numbers(r)

    example = week(replace(BASE, seed=7))["days"]
    tables = {
        "example_daily": example_table(example),
        "toxicity": toxicity_table(r["toxicity"]),
        "width": width_table(r),
        "lean": lean_table(r["leans"]),
    }

    out = {"base": clean(r["base"]), "derived": derived}
    out["tables"] = tables
    out["toxicity"] = [clean(t) for t in r["toxicity"]]
    out["width"] = {
        "no_informed": [clean(w) for w in r["width_calm"]],
        "base": [clean(w) for w in r["width"]],
    }
    out["hedge_band"] = [clean(x) for x in r["bands"]]
    out["lean"] = [clean(x) for x in r["leans"]]
    return out


def clean(s):
    """A setting without its per-week arrays."""
    return {x: v for x, v in s.items() if x != "per_week"}


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


def chart_day(out):
    """Day one of the example week, minute by minute."""
    run = simulate(replace(BASE, seed=7))
    n = MINUTES
    hours = np.arange(n + 1) / 60
    size = dict(figsize=(7, 9.6), height_ratios=[1, 1, 1, 1.6])
    fig, ax = plt.subplots(4, 1, sharex=True, **size)

    ax[0].plot(hours, run["spot"][: n + 1], color=BLUE)
    ax[1].plot(hours, 100 * run["level"][: n + 1], color=BLUE)
    ax[2].plot(hours[:-1], run["net_vega"][:n], color=BLUE)
    day_pnl(ax[3], hours, run["pnl"])
    ax[3].set_xticks(range(7))
    ax[3].set_xlabel("hours into the session")

    finish(ax[0], "The stock price", money=True)
    finish(ax[1], "The vol level, in vol points")
    finish(ax[2], "Net vega, in dollars per vol point", money=True)
    finish(ax[3], "Cumulative P&L of the day", money=True, legend=True)
    path = out / "day_in_the_life.png"
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def day_pnl(ax, hours, pnl):
    """Cumulative edge, market moves with hedging, and the total."""
    n = len(hours) - 1
    cum = {x: np.r_[0, np.cumsum(v[:n])] for x, v in pnl.items()}
    costs = sum(cum[x] for x in PIECES if x != "edge")
    ax.plot(hours, cum["edge"], color=BLUE, label="edge")
    label = "market moves and hedging"
    ax.plot(hours, costs, color=ORANGE, label=label)
    ax.plot(hours, cum["edge"] + costs, color=GREEN, label="total")


def chart_markouts(res, out):
    """Hedged markout per contract by customer, over the first ten
    minutes and over four hours."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.6))
    early = [h for h in HORIZONS if h <= 10]
    for kd, colour in zip(KINDS, (BLUE, ORANGE, GREEN), strict=True):
        mk = res["base"]["markout"][kd]
        ax[0].plot(
            early, [mk[h] for h in early], color=colour, label=kd
        )
        ax[1].plot(HORIZONS, [mk[h] for h in HORIZONS], color=colour)

    ax[0].set_xticks(range(0, 11, 2))
    ax[1].set_xticks(range(0, 241, 60))
    for a in ax:
        a.set_xlabel("minutes after the fill")

    title = "Hedged markout per contract in the first ten minutes"
    finish(ax[0], title, money=True, legend=True)
    finish(ax[1], "The same markouts over four hours", money=True)
    fig.savefig(out / "markouts.png", metadata={"Software": None})
    plt.close(fig)


def chart_width(res, out):
    """Daily P&L at each half-width, with and without informed flow."""
    fig, ax = plt.subplots()
    lines = (
        ("no_informed", BLUE, "uninformed customers only"),
        ("base", ORANGE, "with informed traders"),
    )

    # a true vol-point axis, so the slopes either side compare
    for key, colour, label in lines:
        totals = [w["total"] for w in res["width"][key]]
        ax.plot(WIDTHS, totals, color=colour, marker="o", label=label)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("half-width of the quote, in vol points")

    title = "Daily P&L against the half-width of the quote"
    finish(ax, title, money=True, legend=True)
    fig.savefig(out / "width.png", metadata={"Software": None})
    plt.close(fig)


def chart_bands(res, out):
    """Hedging cost and the swing of daily P&L for each hedge band."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    where = range(len(BANDS))
    cost = [-x["hedging"] for x in res["hedge_band"]]
    swing = [x["daily_std"] for x in res["hedge_band"]]
    ax[0].plot(where, cost, color=BLUE, marker="o")
    ax[1].plot(where, swing, color=ORANGE, marker="o")
    ax[1].set_xticks(where, [f"{b:,}" for b in BANDS])
    ax[1].set_xlabel("hedge band, in shares of delta tolerated")

    finish(ax[0], "Hedging cost per day", money=True)
    finish(ax[1], "Standard deviation of daily P&L", money=True)
    fig.savefig(out / "hedge_band.png", metadata={"Software": None})
    plt.close(fig)


def main(out=Path(".")):
    res = results()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    chart_day(figures)
    chart_markouts(res, figures)
    chart_width(res, figures)
    chart_bands(res, figures)

    tables = res["tables"]
    text = json.dumps(res, indent=1, default=float) + "\n"
    (out / "results.json").write_text(text)
    print(tables["example_daily"], tables["toxicity"], sep="\n\n")


if __name__ == "__main__":
    main()
