# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIMIT, CANCEL, MARKET = 0, 1, 2
BID, ASK = 0, 1


def rates(p, q):
    """Limit orders, cancellations and market orders per second at a
    best queue holding q lots."""
    limit = p.limit_rate if q < p.max_queue else 0.0
    cancel = p.cancel_rate * q

    # market orders hit short queues harder than long ones
    boost = 1 + p.market_boost * math.exp(-q / p.market_scale)
    return limit, cancel, p.market_rate * boost


def next_event(p, qb, qa, wait, pick):
    """Seconds until the book's next event and which flow it comes
    from, given a unit exponential draw and a uniform draw."""
    flows = rates(p, qb) + rates(p, qa)
    total = sum(flows)
    target = pick * total

    # walk the six flows until their running sum passes the target
    k, running = 0, flows[0]
    while running < target and k < 5:
        k += 1
        running += flows[k]

    side, kind = divmod(k, 3)
    return wait / total, side, kind


def queue_ran_out(p, side, coin_move, coin_news, fresh, deep):
    """Price move in ticks and new bid and ask queues after the best
    queue on `side` runs out. None leaves a queue as it was."""
    if coin_move >= p.theta:
        # a new order refills the level, and the price stays
        return (0, fresh, None) if side == BID else (0, None, fresh)

    move = -1 if side == BID else 1
    if coin_news < p.theta_reinit:
        # news resets the book, and both queues start long
        return move, deep[0], deep[1]

    # the side that ran out gets a long queue, standing for the one
    # that waited a tick behind, and the other side starts short
    if side == BID:
        return move, deep[0], fresh
    return move, fresh, deep[1]


GROUPS = (-2, -1, 0, 1, 2)
NAMES = (
    "strong sell",
    "mild sell",
    "balanced",
    "mild buy",
    "strong buy",
)


def imbalance(qb, qa):
    """Bid queue minus offer queue over their sum, from -1 to 1."""
    return (qb - qa) / (qb + qa)


def group(imb):
    """-2 for a strong sell reading up to 2 for a strong buy reading,
    with the groups mirrored around zero."""
    size = np.abs(imb)
    level = (size >= 0.2).astype(int) + (size >= 0.6).astype(int)
    return np.sign(imb).astype(int) * level


def queue_step(ev, j, side, ahead):
    """What event j does to a resting one-lot order on `side` with
    `ahead` lots in front of it: 'fill', 'away' or 'wait'."""
    move = ev["move"][j]
    if move != 0:
        # a tick down fills a resting bid, a tick up leaves it behind
        toward = move < 0 if side == BID else move > 0
        return ("fill" if toward else "away"), ahead

    if ev["side"][j] != side or ev["kind"][j] == LIMIT:
        return "wait", ahead
    if ev["kind"][j] == MARKET:
        return ("fill", 0) if ahead == 0 else ("wait", ahead - 1)

    # a cancellation comes from a random lot of the queue
    if ev["u"][j] * ev["q_before"][j] < ahead:
        ahead -= 1
    return "wait", ahead


def wait_in_queue(p, ev, i, side):
    """Follow a one-lot order posted after event i at the back of the
    best queue on `side`. Returns whether it filled, and the event at
    which the wait ended."""
    queue = ev["qb"] if side == BID else ev["qa"]
    ahead, j = queue[i], i + 1
    stop = ev["t"][i] + p.deadline

    while j < len(ev["t"]) and ev["t"][j] <= stop:
        status, ahead = queue_step(ev, j, side, ahead)
        if status != "wait":
            return status == "fill", j
        j += 1
    return False, j - 1


def three_routes(p, ev, i, sign):
    """Fill prices in ticks for a signal at event i (sign +1 buys):
    cross now, post and cancel, post and cross on a miss.
    None marks an order that never filled."""
    side = BID if sign > 0 else ASK
    bid = ev["bid"]
    cross = bid[i] + (1 if sign > 0 else 0)
    rest = bid[i] + (0 if sign > 0 else 1)

    filled, end = wait_in_queue(p, ev, i, side)
    if filled:
        return cross, rest, rest
    late = bid[end] + (1 if sign > 0 else 0)
    return cross, None, late


def rest_on_one_side(ev, side, threshold, latency):
    """Fill times and prices of a one-lot order kept at the best quote
    on `side`. With a threshold it steps out while the imbalance
    points against it by more than that. Instructions land `latency`
    seconds after the event that prompted them."""
    sign = 1 if side == BID else -1
    t, queue, best = ev["t"], ev[QUEUE[side]], ev[PRICE[side]]
    resting, ahead, price, pending, fills = False, 0, 0, None, []

    for j in range(1, len(t)):
        # an earlier instruction to rest or to step out lands now
        if pending is not None and t[j] >= pending[1]:
            resting, pending = pending[0], None
            ahead, price = queue[j - 1], best[j - 1]

        if resting:
            status, ahead = queue_step(ev, j, side, ahead)
            if status == "fill":
                fills.append((t[j], price))
            resting = status == "wait"

        # decide from the book as this event left it
        want = threshold is None or sign * ev["imb"][j] >= -threshold
        plan = resting if pending is None else pending[0]
        if want != plan:
            pending = (want, t[j] + latency)
    return fills


@dataclass(frozen=True)
class Params:
    # order flow at each best queue, per second
    limit_rate: float = 4.0  # new lots joining the queue
    cancel_rate: float = 0.3  # cancellations per resting lot
    market_rate: float = 1.2  # market orders at a long queue
    market_boost: float = 2.0  # extra market orders at a short one
    market_scale: float = 3.0  # lots over which the boost fades
    max_queue: int = 50  # lots

    # what happens when a best queue runs out
    theta: float = 0.7  # chance that the price moves
    theta_reinit: float = 0.3  # chance that a move resets the book
    depth_fresh: float = 3.0  # mean lots at a level just opened
    depth_level2: float = 10.0  # mean lots at a level that waited

    # the session and the trader
    seconds: float = 3600.0
    events_cap: int = 60  # random draws per second of session
    deadline: float = 10.0  # seconds a posted order waits
    markout_horizon: float = 30.0  # seconds after a signal
    rest_horizon: float = 10.0  # seconds after a resting fill
    threshold: float = 0.6  # imbalance at which a quote steps out


SESSIONS = 40
BASE = Params()
COLUMNS = ("t", "side", "kind", "q_before", "qb", "qa", "bid", "move")
QUEUE, PRICE = ("qb", "qa"), ("bid", "ask")


def draws(p, seed):
    """Every random number one session can use, as Python lists."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    n = int(p.seconds * p.events_cap)
    return {
        "wait": rng.exponential(size=n).tolist(),
        "pick": rng.random(n).tolist(),
        "coins": rng.random((n, 2)).tolist(),
        "fresh": rng.geometric(1 / p.depth_fresh, n).tolist(),
        "deep": rng.geometric(1 / p.depth_level2, (n, 2)).tolist(),
        "u": rng.random(n).tolist(),
    }


def session(p, seed):
    """One simulated session, as arrays with one row per event."""
    d = draws(p, seed)
    qb, qa = d["deep"][-1]
    t, bid, rows = 0.0, 0, [(0.0, -1, -1, 0, qb, qa, 0, 0)]

    for i in range(len(d["wait"])):
        wait, side, kind = next_event(
            p, qb, qa, d["wait"][i], d["pick"][i]
        )
        t += wait
        if t > p.seconds:
            break
        before = qb if side == BID else qa
        after = before + 1 if kind == LIMIT else before - 1
        qb, qa = (after, qa) if side == BID else (qb, after)

        move = 0
        if after == 0:
            draw = (*d["coins"][i], d["fresh"][i], d["deep"][i])
            move, new_b, new_a = queue_ran_out(p, side, *draw)
            qb = qb if new_b is None else new_b
            qa = qa if new_a is None else new_a
            bid += move
        rows.append((t, side, kind, before, qb, qa, bid, move))
    else:
        raise RuntimeError("the session ran out of random draws")

    log = {c: np.array(v) for c, v in zip(COLUMNS, zip(*rows))}
    return log | {"u": np.array([0.0, *d["u"][: len(rows) - 1]])}


def prepare(log):
    """Add the offer and the imbalance to a session's log, and return
    the log with a copy of every column as a list."""
    log["ask"] = log["bid"] + 1
    log["imb"] = imbalance(log["qb"], log["qa"])
    return log, {c: v.tolist() for c, v in log.items()}


def mid_at(log, times):
    """The mid-price in ticks at each of the given times."""
    j = np.searchsorted(log["t"], times, side="right") - 1
    return log["bid"][j] + 0.5


HORIZONS = (0.5, 1, 2, 5, 10, 30, 60, 120)


def readings(p, log):
    """The book on a one-second clock, leaving room for the longest
    horizon at the end of the session."""
    grid = np.arange(1.0, p.seconds - max(HORIZONS) + 0.5)
    i = np.searchsorted(log["t"], grid, side="right") - 1
    mid = log["bid"][i] + 0.5
    moves = np.array([mid_at(log, grid + h) - mid for h in HORIZONS])

    # the first price change after each reading, if there is one
    changes = np.flatnonzero(log["move"])
    k = np.searchsorted(changes, i, side="right")
    has_next = k < len(changes)
    up = log["move"][changes[np.minimum(k, len(changes) - 1)]] > 0

    g = group(log["imb"][i])
    return dict(
        grid=grid, i=i, group=g, moves=moves, up=up, ok=has_next
    )


def by_group(r):
    """Share of time, chance of an up move and mean move by group."""
    share, p_up, move = [], [], []
    for g in GROUPS:
        sel = r["group"] == g
        share.append(sel.mean())
        nxt = sel & r["ok"]
        p_up.append(r["up"][nxt].mean() if nxt.any() else np.nan)
        empty = np.full(len(HORIZONS), np.nan)
        move.append(r["moves"][:, sel].mean(1) if sel.any() else empty)
    return {"share": share, "p_up": p_up, "move": move}


def book_numbers(p, log, r):
    """Event rate, queue size and price changes in one session."""
    changes = log["move"][log["move"] != 0]
    queues = log["qb"][r["i"]] + log["qa"][r["i"]]
    return {
        "event_rate": (len(log["t"]) - 1) / p.seconds,
        "mean_queue": queues.mean() / 2,
        "seconds_per_change": p.seconds / len(changes),
        "alternation": (changes[1:] != changes[:-1]).mean(),
        "all_move": r["moves"].mean(1),
    }


CURVE = (0, 1, 2, 5, 10, 20, 30, 60)


def route_rows(p, log, ev, r):
    """One row per strong signal, and the passive fills for the
    markout curves."""
    g, later = r["group"], mid_at(log, r["grid"] + p.markout_horizon)
    entered = (np.abs(g) == 2) & (g != np.r_[0, g[:-1]])
    rows, fills = [], {"signal": [], "random": []}

    for k in range(len(g)):
        i, mid = r["i"][k], log["bid"][r["i"][k]] + 0.5
        for sign in (1, -1):
            if k % 10 == 0:
                fills["random"] += passive_fill(p, ev, i, sign)
            if not (entered[k] and np.sign(g[k]) == sign):
                continue

            # the three routes, the queue ahead and the move missed
            prices = three_routes(p, ev, i, sign)
            values = [value(x, later[k], sign) for x in prices]
            ahead = ev[QUEUE[0 if sign > 0 else 1]][i]
            missed = sign * (later[k] - mid)
            rows.append([*values, prices[1] is not None, ahead, missed])
            fills["signal"] += passive_fill(p, ev, i, sign)
    return np.array(rows, float), fills


def value(price, mid_later, sign):
    """Markout of a fill in ticks, or zero for no fill."""
    return 0.0 if price is None else sign * (mid_later - price)


def passive_fill(p, ev, i, sign):
    """A list with the fill time, price and sign of a passive order
    posted after event i, or an empty list if it did not fill."""
    filled, end = wait_in_queue(p, ev, i, BID if sign > 0 else ASK)
    price = ev["bid"][i] + (0 if sign > 0 else 1)
    return [(ev["t"][end], price, sign)] if filled else []


def route_numbers(p, log, ev, r):
    """Per-session averages for the three routes and the curves."""
    rows, fills = route_rows(p, log, ev, r)
    filled = rows[:, 3] == 1
    out = {"values": rows[:, :3].mean(0), "fill_rate": filled.mean()}
    out["post_markout"] = rows[filled, 1].mean()
    out["ahead"] = rows[:, 4].mean()
    out["missed"] = rows[~filled, 5].mean()

    # markout curves from each passive fill's own time and price
    for name, kept in fills.items():
        t, price, sign = (np.array(c, float) for c in zip(*kept))
        later = [sign * (mid_at(log, t + h) - price) for h in CURVE]
        out[f"curve_{name}"] = np.mean(later, axis=1)

    # random orders were posted at every tenth reading, on both sides
    posted = 2 * len(range(0, len(r["grid"]), 10))
    out["random_fill_rate"] = len(fills["random"]) / posted
    return out


LATENCIES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def resting_pnl(p, log, ev, threshold, latency):
    """Fills and markout in ticks of the resting bid and offer."""
    fills, total = 0, 0.0
    for side, sign in ((BID, 1), (ASK, -1)):
        done = rest_on_one_side(ev, side, threshold, latency)
        t = np.array([f[0] for f in done])
        t = t[t + p.rest_horizon <= p.seconds]
        price = np.array([f[1] for f in done])[: len(t)]
        later = mid_at(log, t + p.rest_horizon)
        fills += len(t)
        total += float(np.sum(sign * (later - price)))
    return fills, total


def resting_numbers(p, log, ev):
    """Fills and markout an hour for both settings at each latency."""
    hours = p.seconds / 3600
    out = {"none": [], "filter": []}
    for lat in LATENCIES:
        for name, thr in (("none", None), ("filter", p.threshold)):
            fills, total = resting_pnl(p, log, ev, thr, lat)
            out[name].append((fills / hours, total / hours))
    return out


def run_session(p, seed):
    """Every measurement the article makes on one session."""
    log, ev = prepare(session(p, seed))
    r = readings(p, log)
    return {
        "book": book_numbers(p, log, r),
        "groups": by_group(r),
        "routes": route_numbers(p, log, ev, r),
        "resting": resting_numbers(p, log, ev),
    }


def mean_se(x):
    """Mean over sessions and its standard error, skipping sessions
    where a value is missing."""
    x = np.asarray(x, float)
    n = np.sum(~np.isnan(x), axis=0)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(n)
    return np.nanmean(x, axis=0), se


def stack(runs, part, key):
    """One field of one part of every session, as an array."""
    return np.array([run[part][key] for run in runs], float)


def summary(runs):
    """Means and standard errors of everything, by session."""
    out = {}
    for part in ("book", "groups", "routes"):
        for key in runs[0][part]:
            m, se = mean_se(stack(runs, part, key))
            out[key], out[f"{key}_se"] = m, se

    rest = {k: stack(runs, "resting", k) for k in ("none", "filter")}
    gain = rest["filter"][:, :, 1] - rest["none"][:, :, 1]
    out["gain"], out["gain_se"] = mean_se(gain)
    for k in ("none", "filter"):
        out[f"{k}_fills"], _ = mean_se(rest[k][:, :, 0])
        out[f"{k}_pnl"], _ = mean_se(rest[k][:, :, 1])
    return out, stack(runs, "routes", "values")


COPIED = ("event_rate", "mean_queue", "seconds_per_change")


def book_text(s):
    """Numbers for the sections on the book and its price changes."""
    out = {k: s[k] for k in COPIED}
    out["sessions"] = SESSIONS
    out["alternation_pct"] = 100 * s["alternation"]
    out["alternation_se_pct"] = 100 * s["alternation_se"]

    # the zero checks of the signal section
    zero_z = np.abs(s["all_move"]) / s["all_move_se"]
    out["zero_move_max_z"] = zero_z.max()
    gap = abs(s["p_up"][0] + s["p_up"][4] - 1)
    out["mirror_gap_pct"] = 100 * gap
    return out


def signal_text(s):
    """Numbers for the sections on the imbalance and the moves."""
    strong, se = s["move"][4], s["move_se"][4]
    one, ten, sixty = (HORIZONS.index(h) for h in (1, 10, 60))
    out = {"p_up_strong_buy_pct": 100 * s["p_up"][4]}
    out["p_up_balanced_pct"] = 100 * s["p_up"][2]
    out["share_strong_pct"] = 100 * (s["share"][0] + s["share"][4])

    out["move_1_strong"] = strong[one]
    out["move_10_strong"] = strong[ten]
    out["move_60_strong"], out["move_60_se"] = strong[sixty], se[sixty]
    out["plateau_share_of_half_pct"] = 100 * strong[sixty] / 0.5
    return out


def round_trip_text(s):
    """The best aggressive round trip, and buying at every reading."""
    signs = np.sign(np.array(GROUPS))[:, None]
    trips = signs * s["move"] - 1
    trips[2] = -np.inf  # a balanced reading gives no direction
    g, h = np.unravel_index(np.argmax(trips), trips.shape)

    out = {"best_round_trip_loss": -trips[g, h]}
    out["best_round_trip_se"] = s["move_se"][g, h]

    # buying at every reading and selling at the longest horizon
    out["random_round_trip_loss"] = 1 - s["all_move"][-1]
    out["random_round_trip_se"] = s["all_move_se"][-1]
    return out


def route_text(p, s, values):
    """Numbers for the section on posting instead of crossing."""
    out = {"deadline": p.deadline, "markout_horizon": p.markout_horizon}
    pairs = {"post_minus_cross": 0, "post_minus_ptc": 2}
    for name, other in pairs.items():
        diff = mean_se(values[:, 1] - values[:, other])
        out[name], out[f"{name}_se"] = diff

    out["post_fill_pct"] = 100 * s["fill_rate"]
    out["post_markout"] = s["post_markout"]
    out["queue_at_signal"] = s["ahead"]
    out["queue_mean"] = s["mean_queue"]
    out["missed_move"] = s["missed"]

    thirty = CURVE.index(p.markout_horizon)
    out["random_fill_pct"] = 100 * s["random_fill_rate"]
    out["curve_random_30"] = s["curve_random"][thirty]
    out["curve_signal_30"] = s["curve_signal"][thirty]
    return out


def resting_text(p, s):
    """Numbers for the section on stepping out of the way."""
    out = {"threshold": p.threshold, "rest_horizon": p.rest_horizon}
    none_fills, none_pnl = s["none_fills"][0], s["none_pnl"][0]
    filter_fills, filter_pnl = s["filter_fills"][0], s["filter_pnl"][0]
    out["none_fills"], out["none_pnl"] = none_fills, none_pnl
    out["none_markout"] = none_pnl / none_fills
    out["filter_markout"] = filter_pnl / filter_fills
    out["fills_lost_pct"] = 100 * (1 - filter_fills / none_fills)
    out["none_pnl_last"] = s["none_pnl"][-1]

    # the filter's loss against no filter at three of the latencies
    for name, lat in (("0", 0.0), ("50", 0.05), ("mid", 0.25)):
        k = LATENCIES.index(lat)
        out[f"filter_loss_{name}"] = -s["gain"][k]
        out[f"filter_loss_{name}_se"] = s["gain_se"][k]
        out[f"lat_{name}_events"] = s["event_rate"] * lat
        out[f"lat_{name}_ms"] = 1000 * lat
    return out


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def groups_table(s):
    """Share of time, chance of an up move and moves, by group."""
    one, ten = HORIZONS.index(1), HORIZONS.index(10)
    rows = []
    for k, name in enumerate(NAMES):
        share = f"{100 * s['share'][k]:.0f}%"
        p_up = f"{100 * s['p_up'][k]:.0f}%"
        moves = [f"{s['move'][k][h]:+.2f}" for h in (one, ten)]
        rows.append([name, share, p_up, *moves])

    header = ["imbalance", "time", "next up", "move 1s", "move 10s"]
    return table(rows, header, [11, 4, 7, 7, 8])


def routes_table(s):
    """Fill rate, markout per fill and per signal, and its standard
    error."""
    names = ("cross now", "post, cancel", "post, then cross")
    fill = [1.0, s["fill_rate"], 1.0]
    rows = []
    for k, name in enumerate(names):
        per_fill = s["values"][k] / fill[k]
        cells = [f"{100 * fill[k]:.0f}%", f"{per_fill:+.2f}"]
        cells += [f"{s['values'][k]:+.2f}", f"{s['values_se'][k]:.2f}"]
        rows.append([name, *cells])

    header = ["route", "filled", "per fill", "per signal", "std error"]
    return table(rows, header, [16, 6, 8, 10, 9])


def results():
    """Every number and table the article quotes, ready for JSON."""
    runs = [run_session(BASE, seed) for seed in range(SESSIONS)]
    s, values = summary(runs)
    derived = book_text(s) | signal_text(s) | round_trip_text(s)
    derived |= route_text(BASE, s, values) | resting_text(BASE, s)

    tables = {"groups": groups_table(s), "routes": routes_table(s)}
    cover = {
        "strong_buy": s["move"][4].tolist(),
        "balanced": s["move"][2].tolist(),
        "half_spread": [0.5] * len(HORIZONS),
    }
    out = {"derived": derived, "tables": tables, "cover": cover}
    out["series"] = {k: np.asarray(v).tolist() for k, v in s.items()}
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


def chart_p_up(s, out):
    """The chance of an up move by group, with two standard errors."""
    fig, ax = plt.subplots()
    middles = [-0.8, -0.4, 0.0, 0.4, 0.8]
    p_up = 100 * np.array(s["p_up"])
    bars = 200 * np.array(s["p_up_se"])
    ax.errorbar(middles, p_up, bars, color=BLUE, marker="o", capsize=3)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("queue imbalance, bid minus offer over their sum")
    title = "Chance that the next price move is up, in percent"
    finish(ax, title)
    save(fig, out / "p_up.png")


def chart_moves(s, out):
    """Average move of the mid after each kind of reading."""
    fig, ax = plt.subplots()
    where = range(len(HORIZONS))
    for g, colour in ((4, BLUE), (2, ORANGE), (0, GREEN)):
        style = dict(color=colour, marker="o", label=NAMES[g])
        ax.plot(where, s["move"][g], **style)

    # the half-spread and the full spread, labelled at the right end
    lines = ((0.5, "half-spread", "bottom"), (1.0, "round trip", "top"))
    for level, name, va in lines:
        ax.axhline(level, color=GREY, lw=0.9)
        y = level + (0.04 if va == "bottom" else -0.04)
        style = dict(color=GREY, ha="right", va=va)
        ax.text(where[-1], y, name, **style)

    ax.set_xticks(where, [f"{h:g}" for h in HORIZONS])
    ax.set_xlabel("seconds after the reading")
    finish(ax, "Average move of the mid, in ticks", legend=True)
    save(fig, out / "moves.png")


def chart_markouts(s, out):
    """Markout after passive fills, on the signal and at random."""
    fig, ax = plt.subplots()
    lines = (("curve_signal", BLUE, "posted on a strong signal"),)
    lines += (("curve_random", ORANGE, "posted at random times"),)
    for key, colour, label in lines:
        ax.plot(CURVE, s[key], color=colour, marker="o", label=label)

    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("seconds after the fill")
    title = "Markout after a passive fill, in ticks"
    finish(ax, title, legend=True)
    save(fig, out / "markouts.png")


def chart_latency(s, out):
    """The filter's gain in markout an hour at each latency."""
    fig, ax = plt.subplots()
    where = range(len(LATENCIES))
    bars = 2 * np.array(s["gain_se"])
    style = dict(color=BLUE, marker="o", capsize=3)
    ax.errorbar(where, s["gain"], bars, **style)

    labels = [f"{1000 * x:,.0f}" for x in LATENCIES]
    ax.set_xticks(where, labels)
    ax.set_xlabel("latency, in milliseconds")
    finish(ax, "Gain from the filter, in ticks of markout an hour")
    save(fig, out / "latency.png")


def main(out=Path(".")):
    res = results()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    s = res["series"]
    chart_p_up(s, figures)
    chart_moves(s, figures)
    chart_markouts(s, figures)
    chart_latency(s, figures)

    text = json.dumps(res, indent=1, default=float) + "\n"
    (out / "results.json").write_text(text)
    print(res["tables"]["groups"], res["tables"]["routes"], sep="\n\n")


if __name__ == "__main__":
    main()
