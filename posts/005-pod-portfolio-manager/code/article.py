# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

YEAR = 252  # trading days in a year


def daily_return(ir, vol, z):
    """A day's P&L on a full allocation, as a share of it.

    The manager earns ir times vol a year on average, and the
    daily shock z, which has unit variance, adds vol of noise a year.
    """
    return ir * vol / YEAR + vol / math.sqrt(YEAR) * z


def apply_rule(p, pnl, peak, scale, seated):
    """The drawdown rule, checked at the end of each day.

    pnl is cumulative P&L and peak its high-water mark, both as a
    share of the original allocation. Returns the new high-water
    mark, the allocation's scale and who still has the job.
    """
    peak = np.maximum(peak, pnl)
    drawdown = peak - pnl
    seated = seated & (drawdown < p.stop)

    # halve the book at the cut, and restore it at the next new high
    cut = seated & (scale == 1) & (drawdown >= p.cut)
    back = (scale < 1) & (drawdown <= p.restore)
    scale = np.where(back, 1.0, scale)
    scale = np.where(cut, p.cut_scale, scale)
    return peak, scale, seated


def year_end(p, year_pnl, owed, seated):
    """The payout at a year end, with losses carried forward.

    owed is the loss still to be won back before payouts resume.
    A manager let go during the year is paid nothing for it.
    """
    payout = p.payout_rate * np.maximum(0, year_pnl - owed) * seated
    owed = np.maximum(0, owed - year_pnl)
    return payout, owed


def careers(p, n=None, seed=None):
    """n managers of skill p.ir, each run for p.years under p's rule.

    For a given seed every setting meets the same daily shocks.
    """
    n = n or PMS
    rng = np.random.default_rng(SEED if seed is None else seed)
    pnl, peak, year_pnl, owed, paid = (np.zeros(n) for _ in range(5))
    scale, seated = np.ones(n), np.ones(n, bool)
    tenure, cut_once = np.zeros(n), np.zeros(n, bool)

    for day in range(1, p.years * YEAR + 1):
        z = shocks(p, rng, n)
        today = scale * seated * daily_return(p.ir, p.vol, z)
        pnl += today
        year_pnl += today
        peak, scale, seated = apply_rule(p, pnl, peak, scale, seated)
        cut_once |= scale < 1
        tenure += seated

        if day % YEAR == 0:
            payout, owed = year_end(p, year_pnl, owed, seated)
            paid += payout
            year_pnl[:] = 0

    out = dict(tenure=tenure, cut=cut_once, paid=paid, pnl=pnl)
    return out | {"peak": peak}


def mean_years_to_stop(ir, vol, stop):
    """Mean years until a drawdown of `stop`, with no cut and no
    horizon, for a manager who never changes risk."""
    if ir == 0:
        return stop**2 / vol**2

    # the high-water mark at the stop is exponential with this mean
    a = 2 * ir / vol
    peak = (math.exp(a * stop) - 1) / a
    return (peak - stop) / (ir * vol)


def refill(p, s, rec, let_go):
    """Empty seats wait p.idle_days, then take the slot's next
    candidate with a full allocation and a fresh high-water mark."""
    s["idle"] = np.where(let_go, p.idle_days, s["idle"] - 1)
    new = ~s["seated"] & (s["idle"] <= 0)
    s["hire"] = np.minimum(s["hire"] + new, p.max_hires - 1)
    s["seated"] = s["seated"] | new

    for x in ("pnl", "peak", "year", "owed", "days"):
        s[x] = np.where(new, 0.0, s[x])
    s["scale"] = np.where(new, 1.0, s["scale"])
    rec["hires"] += new.sum(axis=1)


@dataclass(frozen=True)
class Params:
    # the manager
    ir: float = 1.0  # information ratio, edge per unit of risk
    vol: float = 0.06  # yearly volatility on the allocation
    tails: bool = False  # fat-tailed days instead of normal ones

    # the drawdown rule, as shares of the allocation
    cut: float = 0.05  # drawdown that halves the allocation
    stop: float = 0.075  # drawdown that ends the job
    cut_scale: float = 0.5  # the allocation after a cut
    restore: float = 0.0  # drawdown at which a cut is undone

    # the pay and the clock
    payout_rate: float = 0.20  # share of each year's gain paid out
    years: int = 5

    # the platform
    pods: int = 50
    runs: int = 400
    rho: float = 0.1  # correlation of daily shocks across pods
    idle_days: int = 63  # a seat stays empty this long after a stop
    pool: tuple = (0.0, 0.5, 1.0, 1.5)  # skill of possible hires
    odds: tuple = (0.4, 0.3, 0.2, 0.1)
    fade_years: float = 0.0  # mean life of an edge, 0 for forever
    max_hires: int = 16


BASE = Params()
PMS, SEED = 10_000, 2026
INF = math.inf

RULES = {
    "none": (INF, INF),
    "stop only": (INF, 0.075),
    "base": (0.05, 0.075),
    "tight": (0.03, 0.05),
    "loose": (0.10, 0.15),
}
IRS = (0.0, 0.5, 1.0, 1.5, 2.0)
SHOWN = (0.0, 1.0, 2.0)
VOLS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
WEAK_POOL = (0.6, 0.25, 0.1, 0.05)


def rule(name, p=BASE):
    """p with one of the named drawdown rules."""
    cut, stop = RULES[name]
    return replace(p, cut=cut, stop=stop)


def shocks(p, rng, shape):
    """Daily shocks with unit variance, normal or Student t."""
    if p.tails:  # four degrees of freedom, scaled to unit variance
        return rng.standard_t(4, shape) / math.sqrt(2)
    return rng.standard_normal(shape)


_careers = {}


def career(p):
    """careers(p) on the article's seed, run once and kept."""
    if p not in _careers:
        _careers[p] = careers(p)
    return _careers[p]


def mean_se(x):
    """Mean and standard error of independent values."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def paired(a, b):
    """Mean and standard error of b minus a, unit by unit."""
    return mean_se(np.asarray(b) - np.asarray(a))


def summary(p):
    """What the article reports about one single-manager setting."""
    c = career(p)
    n, days = len(c["tenure"]), p.years * YEAR
    out = {"cut_once": float(c["cut"].mean())}
    for y in range(1, p.years + 1):
        share = float((c["tenure"] >= y * YEAR).mean())
        out[f"employed_{y}"] = share
        out[f"employed_{y}_se"] = math.sqrt(share * (1 - share) / n)

    out["paid"], out["paid_se"] = mean_se(c["paid"])
    out["take"], out["take_se"] = mean_se(c["pnl"] - c["paid"])
    out["pnl"], out["pnl_se"] = mean_se(c["pnl"])
    monthly = range(0, days + 1, 21)
    out["survival"] = [
        float((c["tenure"] >= d).mean()) for d in monthly
    ]
    return out


def candidates(p, rng, shape):
    """The skill of each future hire, and the day in the seat on
    which that hire's edge fades to nothing."""
    skill = rng.choice(p.pool, size=shape, p=p.odds)
    fade = np.full(shape, INF)
    if p.fade_years:
        fade = rng.exponential(p.fade_years * YEAR, shape)
    return skill, fade


def pod_shocks(p, rng):
    """Each pod's shock, with a common part that every pod shares."""
    common = shocks(p, rng, (p.runs, 1))
    own = shocks(p, rng, (p.runs, p.pods))
    return math.sqrt(p.rho) * common + math.sqrt(1 - p.rho) * own


def platform_day(p, s, rec, skill, fade, rng, day):
    """P&L, the rule, payouts and hiring for one day."""
    k = s["hire"][..., None]
    ir = np.take_along_axis(skill, k, 2)[..., 0]
    gone = s["days"] >= np.take_along_axis(fade, k, 2)[..., 0]
    edge = np.where(gone, 0.0, ir)
    if day in CHECKS:
        rec["bench"].append(ir[s["seated"]].mean())

    was = s["seated"]
    z = pod_shocks(p, rng)
    today = s["scale"] * was * daily_return(edge, p.vol, z)
    s["pnl"] += today
    s["year"] += today
    s["days"] += was
    rec["gross"][:, day - 1] = today.sum(axis=1)
    track_first(rec, s, today, day)

    rule_out = apply_rule(p, s["pnl"], s["peak"], s["scale"], was)
    s["peak"], s["scale"], s["seated"] = rule_out
    let_go = was & ~s["seated"]
    rec["let_go"] += let_go.sum(axis=1)
    first_out = let_go & (s["hire"] == 0)
    rec["first_out"] = np.where(first_out, day, rec["first_out"])

    if day % YEAR == 0:
        pay_year(p, s, rec, day)
    refill(p, s, rec, let_go)


CHECKS = (1, YEAR + 1, 3 * YEAR + 1, 5 * YEAR)


def pay_year(p, s, rec, day):
    """Pay every seated manager for the year that just ended."""
    payout, s["owed"] = year_end(p, s["year"], s["owed"], s["seated"])
    rec["payouts"][:, day // YEAR - 1] = payout.sum(axis=1)
    rec["first_paid"] += payout * (s["hire"] == 0)
    s["year"][:] = 0.0


def track_first(rec, s, today, day):
    """Sums for the first hire's Sharpe ratio over three years."""
    first = (s["hire"] == 0) & (s["days"] > 0) & (today != 0)
    if day <= 3 * YEAR:
        rec["n"] += first
        rec["sum"] += today * first
        rec["sq"] += today**2 * first


_platforms = {}


def platform(p, seed=None):
    """p.runs platforms of p.pods seats each, over p.years."""
    key = (p, seed)
    if key in _platforms:
        return _platforms[key]
    seeds = np.random.SeedSequence(SEED if seed is None else seed)
    people, market = (np.random.default_rng(x) for x in seeds.spawn(2))
    shape = (p.runs, p.pods)
    skill, fade = candidates(p, people, (*shape, p.max_hires))

    s = {x: np.zeros(shape) for x in ("pnl", "peak", "year", "owed")}
    s |= {"days": np.zeros(shape), "idle": np.zeros(shape)}
    s |= {"scale": np.ones(shape), "seated": np.ones(shape, bool)}
    s["hire"] = np.zeros(shape, int)
    rec = {x: np.zeros(shape) for x in ("first_paid", "n", "sum", "sq")}
    rec["first_out"] = np.full(shape, INF)
    rec["gross"] = np.zeros((p.runs, p.years * YEAR))
    rec["payouts"] = np.zeros((p.runs, p.years))
    rec |= {"hires": np.full(p.runs, p.pods), "let_go": 0, "bench": []}

    for day in range(1, p.years * YEAR + 1):
        platform_day(p, s, rec, skill, fade, market, day)
    _platforms[key] = rec | {"skill": skill[..., 0]}
    return _platforms[key]


def fund(p, seed=None):
    """The fund's yearly results and the first hires' payouts, as
    arrays with one value per platform run."""
    rec = platform(p, seed)
    capital = p.pods
    gross = rec["gross"].reshape(p.runs, p.years, YEAR).sum(axis=2)
    net = gross - rec["payouts"]
    vol = rec["gross"].std(axis=1) * math.sqrt(YEAR)

    out = {
        "gross": gross.mean(axis=1) / capital,
        "payouts": rec["payouts"].mean(axis=1) / capital,
        "net": net.mean(axis=1) / capital,
        "sharpe": net.mean(axis=1) / vol,
        "loss_years": (net < 0).mean(axis=1),
        "let_go": rec["let_go"] / rec["hires"],
    }
    for ir in (0.0, 1.0):
        mine = np.where(rec["skill"] == ir, rec["first_paid"], np.nan)
        out[f"paid_ir{ir:.0f}"] = np.nanmean(mine, axis=1)
    return out


def single_runs():
    """Survival by skill and rule, and pay against risk."""
    r = {key: {} for key in ("survival", "risk", "free")}
    for ir in IRS:
        p = replace(BASE, ir=ir)
        for name in ("base", "stop only", "none"):
            r["survival"][(ir, name)] = summary(rule(name, p))
        for vol in (0.04, 0.10):
            r["survival"][(ir, vol)] = summary(replace(p, vol=vol))

        # pay against risk, with the base rule and with no rule
        for vol in VOLS:
            q = replace(p, vol=vol)
            r["risk"][(ir, vol)] = summary(q)
            r["free"][(ir, vol)] = summary(rule("none", q))
    return r


def platform_runs():
    """Every rule, with lasting and fading edge, and the variants."""
    fading = replace(BASE, fade_years=3.0)
    r = {key: {} for key in ("lasting", "fading", "variants")}
    for name in RULES:
        r["lasting"][name] = fund(rule(name))
        r["fading"][name] = fund(rule(name, fading))

    variants = {
        "correlation 0.3": replace(BASE, rho=0.3),
        "weaker pool": replace(BASE, odds=WEAK_POOL),
        "fat tails": replace(BASE, tails=True),
    }
    for label, p in variants.items():
        base, none = fund(rule("base", p)), fund(rule("none", p))
        r["variants"][label] = paired(none["net"], base["net"])
    return r


def survival_numbers(r):
    """Numbers for the section on how long a manager lasts."""
    s, out = r["survival"], {}
    for ir in IRS:
        tag = f"ir{10 * ir:.0f}"
        base = s[(ir, "base")]
        for y in (1, 3, 5):
            out[f"{tag}_employed_{y}"] = 100 * base[f"employed_{y}"]
        out[f"{tag}_cut_once"] = 100 * base["cut_once"]

        # five years with a stop and no cut, and at other vols
        only = s[(ir, "stop only")]
        out[f"{tag}_stop_only_5"] = 100 * only["employed_5"]
        out[f"{tag}_vol4_5"] = 100 * s[(ir, 0.04)]["employed_5"]
        out[f"{tag}_vol10_5"] = 100 * s[(ir, 0.10)]["employed_5"]
        years = mean_years_to_stop(ir, BASE.vol, BASE.stop)
        out[f"{tag}_mean_stop_years"] = years

    # how far the no-skill mean lands from zero, in standard errors
    for name in ("base", "none"):
        zero = s[(0.0, name)]
        out[f"ir0_zero_{name}"] = 100 * zero["pnl"]
        out[f"ir0_zero_{name}_z"] = abs(zero["pnl"]) / zero["pnl_se"]

    out["stop_in_vols"] = BASE.stop / BASE.vol
    out["rho"], out["vol_pct"] = BASE.rho, 100 * BASE.vol
    out["payout_rate_pct"] = 100 * BASE.payout_rate
    return out


def risk_numbers(r):
    """Numbers for the section on how much risk to run."""
    out = {}
    for ir in IRS:
        tag = f"ir{10 * ir:.0f}"
        paid = [r["risk"][(ir, v)]["paid"] for v in VOLS]
        take = [r["risk"][(ir, v)]["take"] for v in VOLS]
        best = int(np.argmax(paid))
        out[f"{tag}_best_vol_pm"] = 100 * VOLS[best]
        out[f"{tag}_best_vol_fund"] = 100 * VOLS[int(np.argmax(take))]
        out[f"{tag}_best_paid"] = 100 * paid[best]
        out[f"{tag}_free_paid_15"] = 100 * r["free"][(ir, 0.15)]["paid"]

        # the payout at the best vol against each neighbour
        top = career(rule("base", replace(BASE, ir=ir, vol=VOLS[best])))
        for side, j in (("below", best - 1), ("above", best + 1)):
            if 0 <= j < len(VOLS):
                q = replace(BASE, ir=ir, vol=VOLS[j])
                d, se = paired(career(q)["paid"], top["paid"])
                out[f"{tag}_best_beats_{side}"] = 100 * d
                out[f"{tag}_best_beats_{side}_se"] = 100 * se
    return out


def record_numbers():
    """Numbers for the section on what a track record shows."""
    rec = platform(BASE)
    skill = rec["skill"].ravel()
    out_day = rec["first_out"].ravel()
    n = rec["n"].ravel()
    mean = rec["sum"].ravel() / np.maximum(n, 1)
    var = rec["sq"].ravel() / np.maximum(n, 1) - mean**2
    sharpe = mean / np.sqrt(np.maximum(var, 1e-18)) * math.sqrt(YEAR)
    kept = out_day > 3 * YEAR
    gone = ~kept & (n >= 21)

    out, rows = {}, []
    for ir in BASE.pool:
        mine = skill == ir
        row = [ir, mine.mean(), 1 - kept[mine].mean()]
        row += [sharpe[mine & kept].mean(), sharpe[mine & gone].mean()]
        rows.append(row)
        tag = f"ir{10 * ir:.0f}"
        out[f"{tag}_let_go_3"] = 100 * (1 - kept[mine].mean())
        out[f"{tag}_sharpe_kept"] = float(sharpe[mine & kept].mean())

    out["skilled_of_let_go"] = 100 * (skill[~kept] >= 1).mean()
    out["skilled_of_hires"] = 100 * (skill >= 1).mean()
    bench = np.array(rec["bench"])
    out["bench_0"], out["bench_5"] = bench[0], bench[-1]
    out["pool_mean"] = float(np.dot(BASE.pool, BASE.odds))
    for sr in (0.0, 1.0):
        lo_se = math.sqrt((1 + sr**2 / (2 * YEAR)) / 3)
        out[f"lo_se_sr{sr:.0f}"] = lo_se
    hist = {ir: sharpe[(skill == ir) & kept] for ir in (0.0, 1.0)}
    return out, rows, hist


def rule_numbers(r):
    """Numbers for the section on who gains from the rule."""
    out = {}
    none = r["lasting"]["none"]
    for name, f in r["lasting"].items():
        tag = name.replace(" ", "_")
        share = f["payouts"].mean() / f["gross"].mean()
        out[f"{tag}_share"] = 100 * share
        out[f"{tag}_sharpe"] = float(f["sharpe"].mean())
        out[f"{tag}_paid_ir1"] = 100 * float(np.nanmean(f["paid_ir1"]))
        fade, fade_none = r["fading"][name], r["fading"]["none"]

        pairs = {
            "net_gain": (none["net"], f["net"]),
            "fade_gain": (fade_none["net"], fade["net"]),
            "ir1_gain": (none["paid_ir1"], f["paid_ir1"]),
        }
        for key, (a, b) in pairs.items():
            d, se = paired(a, b)
            out[f"{tag}_{key}"] = 100 * d
            out[f"{tag}_{key}_se"] = 100 * se
            out[f"{tag}_{key}_neg"] = -100 * d  # a loss, as a size

    for label, (d, se) in r["variants"].items():
        tag = "variant_" + label.replace(" ", "_").replace(".", "")
        out[tag], out[f"{tag}_se"] = 100 * d, 100 * se
        out[f"{tag}_neg"] = -100 * d
    return out


def cut_numbers(r):
    """What the cut does on top of the stop alone."""
    out = {}
    for edge in ("lasting", "fading"):
        only, base = r[edge]["stop only"], r[edge]["base"]
        d, se = paired(only["net"], base["net"])
        out[f"cut_cost_{edge}"] = -100 * d
        out[f"cut_cost_{edge}_se"] = 100 * se

    lasting = r["lasting"]
    sharpes = lasting["stop only"]["sharpe"], lasting["base"]["sharpe"]
    out["cut_sharpe"], out["cut_sharpe_se"] = paired(*sharpes)
    return out


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def survival_table(r):
    """Share still employed by skill, with the formula's stop time."""
    rows = []
    for ir in IRS:
        b = r["survival"][(ir, "base")]
        only = r["survival"][(ir, "stop only")]
        cells = [pct(b[f"employed_{y}"]) for y in (1, 3, 5)]
        cells += [pct(b["cut_once"]), pct(only["employed_5"])]
        years = mean_years_to_stop(ir, BASE.vol, BASE.stop)
        rows.append([f"{ir:.1f}", *cells, f"{years:.1f}"])

    header = ["IR", "1 year", "3 years", "5 years", "cut once"]
    header += ["5y, stop only", "mean years"]
    return table(rows, header, [3, 6, 7, 7, 8, 13, 10])


def record_table(rows):
    """First hires by true skill on the base platform."""
    out = []
    for ir, share, gone, kept_sr, gone_sr in rows:
        cells = [pct(share), pct(gone), f"{kept_sr:.2f}"]
        out.append([f"{ir:.1f}", *cells, f"{gone_sr:.2f}"])

    header = ["true IR", "of hires", "let go in 3y"]
    header += ["Sharpe, kept", "Sharpe, let go"]
    return table(out, header, [7, 8, 12, 12, 14])


def rule_table(r):
    """The fund and the managers under each rule, a year per unit
    of fund capital, except the five-year payout of an IR 1 hire."""
    rows = []
    for name, f in r["lasting"].items():
        share = f["payouts"].mean() / f["gross"].mean()
        cells = [pct(f["gross"].mean(), 2), pct(f["net"].mean(), 2)]
        cells += [pct(share), f"{f['sharpe'].mean():.2f}"]
        cells += [pct(np.nanmean(f["paid_ir1"]), 1)]
        cells += [pct(f["let_go"].mean())]
        cells += [pct(r["fading"][name]["net"].mean(), 2)]
        rows.append([name, *cells])

    header = ["rule", "gross", "net", "paid out", "Sharpe"]
    header += ["IR 1 pay", "let go", "net, fading"]
    return table(rows, header, [9, 5, 5, 8, 6, 8, 6, 11])


def series(single, plat):
    """The lines each chart draws, in percent."""
    out = {key: {} for key in ("cover", "risk", "rules")}
    for ir in SHOWN:
        tag = f"ir{10 * ir:.0f}"
        out["cover"][tag] = single["survival"][(ir, "base")]["survival"]
        for key in ("paid", "take"):
            row = [100 * single["risk"][(ir, v)][key] for v in VOLS]
            out["risk"][f"{key}_{tag}"] = row

    for edge in ("lasting", "fading"):
        nets = {n: 100 * f["net"].mean() for n, f in plat[edge].items()}
        out["rules"][edge] = nets
    paid = {n: f["paid_ir1"] for n, f in plat["lasting"].items()}
    out["rules"]["paid_ir1"] = {
        n: 100 * float(np.nanmean(x)) for n, x in paid.items()
    }
    return out


def results():
    """Every number, table and chart series the article uses."""
    single, plat = single_runs(), platform_runs()
    rec_out, rec_rows, hist = record_numbers()
    derived = survival_numbers(single) | risk_numbers(single)
    derived |= rec_out | rule_numbers(plat) | cut_numbers(plat)

    tables = {
        "survival": survival_table(single),
        "record": record_table(rec_rows),
        "rules": rule_table(plat),
    }
    out = {"derived": derived, "tables": tables}
    out["hist"] = {f"ir{k:.0f}": v.tolist() for k, v in hist.items()}
    return out | series(single, plat)


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


def finish(ax, title, money=False, legend=False, bins=5):
    """Title, round y ticks, a zero line and a legend for a panel."""
    ax.set_title(title, pad=30 if legend else 12)

    lo, hi = ax.dataLim.intervaly
    steps = MaxNLocator(bins, steps=[1, 2, 5, 10])
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


COLOURS = (BLUE, ORANGE, GREEN)


def save(fig, path):
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def chart_survival(res, out):
    """Share of managers still employed, by true skill."""
    fig, ax = plt.subplots()
    for ir, colour in zip(SHOWN, COLOURS, strict=True):
        curve = 100 * np.array(res["cover"][f"ir{10 * ir:.0f}"])
        years = np.arange(len(curve)) * 21 / YEAR
        ax.plot(years, curve, color=colour, label=f"IR {ir:.0f}")

    ax.set_xticks(range(6))
    ax.set_xlim(0, 5)
    ax.set_xlabel("years since the manager was hired")
    title = "Share of managers still employed, in %"
    finish(ax, title, legend=True)
    save(fig, out / "survival.png")


def chart_risk(res, out):
    """Five-year payout and fund take against the volatility run."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)
    vols = 100 * np.array(VOLS)
    for ir, colour in zip(SHOWN, COLOURS, strict=True):
        tag, label = f"ir{10 * ir:.0f}", f"IR {ir:.0f}"
        paid = res["risk"][f"paid_{tag}"]
        ax[0].plot(vols, paid, color=colour, marker="o", label=label)
        take = res["risk"][f"take_{tag}"]
        ax[1].plot(vols, take, color=colour, marker="o")

    ax[1].set_xticks(range(0, 17, 2))
    ax[1].set_xlim(0, 16)
    ax[1].set_xlabel("volatility the manager runs, in % a year")
    title = "Manager's five-year payout, in % of the allocation"
    finish(ax[0], title, legend=True)
    title = "Fund's five-year take, in % of the allocation"
    finish(ax[1], title, bins=8)
    save(fig, out / "risk.png")


def chart_sharpe(res, out):
    """Three-year realised Sharpe ratios of managers who kept the
    job, for true IR 0 and true IR 1."""
    fig, ax = plt.subplots()
    bins = np.arange(-1.5, 4.01, 0.25)
    for ir, colour in zip((0.0, 1.0), (BLUE, ORANGE), strict=True):
        values = np.array(res["hist"][f"ir{ir:.0f}"])
        counts, _ = np.histogram(values, bins)
        share = 100 * counts / len(values)
        label = f"true IR {ir:.0f}"
        steps = np.append(share, share[-1])  # the last bin's top
        ax.step(bins, steps, where="post", color=colour, label=label)

    ax.set_xticks(np.arange(-1, 4.01, 1))
    ax.set_xlim(-1.5, 4)
    ax.set_xlabel("realised Sharpe ratio over the first three years")
    title = "Survivors' three-year Sharpe ratios, in % of each group"
    finish(ax, title, legend=True)
    save(fig, out / "sharpe.png")


EDGES = (
    ("lasting", BLUE, "lasting edge"),
    ("fading", ORANGE, "fading edge"),
)


def chart_rules(res, out):
    """The fund's net P&L and an IR 1 hire's payout under each rule."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)
    names = list(RULES)
    where = range(len(names))
    # points without lines, because the rules are categories
    style = dict(marker="o", linestyle="none", clip_on=False)

    for key, colour, label in EDGES:
        net = [res["rules"][key][n] for n in names]
        ax[0].plot(where, net, color=colour, label=label, **style)

    paid = [res["rules"]["paid_ir1"][n] for n in names]
    ax[1].plot(where, paid, color=BLUE, **style)
    ax[1].set_xticks(where, names)
    ax[1].set_xlabel("drawdown rule")
    title = "Fund's net P&L a year, in % of its capital"
    finish(ax[0], title, legend=True)
    title = "IR 1 hire's five-year payout, in % of the allocation"
    finish(ax[1], title)
    save(fig, out / "rules.png")


CHARTS = (chart_survival, chart_risk, chart_sharpe, chart_rules)


def main(out=Path(".")):
    res = results()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for chart in CHARTS:
        chart(res, figures)

    text = json.dumps(res, indent=1, default=float) + "\n"
    (out / "results.json").write_text(text)
    print(*res["tables"].values(), sep="\n\n")


if __name__ == "__main__":
    main()
