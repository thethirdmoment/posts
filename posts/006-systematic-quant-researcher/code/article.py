# The article's Python blocks, in order, as one file.

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri


def draw_market(p, rng):
    """One world's stock characteristics and next-month returns.

    A stock's return is the planted premium on the true signal, plus
    its exposure to styles that carry no premium, plus its own noise.
    """
    shape = (p.stocks, p.months)
    true = rng.standard_normal(shape)
    styles = rng.standard_normal((p.styles, *shape))

    # fat-tailed style returns with mean zero and sd style_vol
    t = rng.standard_t(p.style_df, (p.styles, 1, p.months))
    unit = t / math.sqrt(p.style_df / (p.style_df - 2))
    style_moves = (styles * p.style_vol * unit).sum(axis=0)

    noise = p.idio_vol * rng.standard_normal(shape)
    returns = premium(p) * true + style_moves + noise
    return true, styles, returns


def candidate_returns(p, rng, true, styles, returns):
    """Monthly long-short returns of every candidate, true one first.

    A candidate holds x / N of each stock, so its return is the
    cross-sectional mean of its signal times next month's return.
    """
    true_ls = (true * returns).mean(axis=0)
    style_ls = (styles * returns).mean(axis=1)

    # each null mixes two styles, with random signs
    nulls = p.candidates - 1
    order = np.argsort(rng.random((nulls, p.styles)), axis=1)
    pair = order[:, :2]
    signs = rng.choice([-1.0, 1.0], (nulls, 2, 1))
    mix = (signs * style_ls[pair]).sum(axis=1) / math.sqrt(2)

    # its own part, drawn from its exact law given the returns
    own_sd = np.sqrt((returns**2).sum(axis=0)) / p.stocks
    own = own_sd * rng.standard_normal((nulls, p.months))
    null_ls = math.sqrt(p.shared) * mix + math.sqrt(1 - p.shared) * own
    return np.vstack([true_ls, null_ls])


def sharpe_stats(x):
    """Monthly Sharpe ratio, t-statistic and one-sided p-value.

    Each row of x is a candidate and each column a month.
    """
    months = x.shape[-1]
    sr = x.mean(axis=-1) / x.std(axis=-1, ddof=1)
    t = sr * math.sqrt(months)
    return sr, t, ndtr(-t)


def bonferroni(pvalues, alpha):
    """Reject where p is below alpha divided by the number of tests."""
    return pvalues <= alpha / len(pvalues)


def holm(pvalues, alpha):
    """Holm's step-down test, which also caps the chance of any
    false discovery, under any dependence between the tests."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    passed = pvalues[order] <= alpha / (m - np.arange(m))
    count = m if passed.all() else int(np.argmin(passed))
    reject = np.zeros(m, bool)
    reject[order[:count]] = True
    return reject


def benjamini_hochberg(pvalues, alpha):
    """The step-up test that caps the false discovery rate."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    below = pvalues[order] <= alpha * np.arange(1, m + 1) / m
    count = below.nonzero()[0].max() + 1 if below.any() else 0
    reject = np.zeros(m, bool)
    reject[order[:count]] = True
    return reject


def walk_forward(p, x, choose):
    """Out-of-sample monthly returns of a way of choosing.

    Every p.step months from p.first_year on, choose(past) sees only
    the months so far and names a candidate, or None for no trade.
    The choice is held for p.step months, sized to unit volatility
    over the past months.
    """
    out, held = [], []

    for start in range(p.first_year, p.months, p.step):
        past = x[:, :start]
        k = choose(past)
        held.append(k)

        if k is None:
            out.append(np.zeros(p.step))
            continue
        size = 1 / past[k].std(ddof=1)
        out.append(size * x[k, start : start + p.step])
    return np.concatenate(out), held


def psr(sr, threshold, months, skew, kurt):
    """Probability that the true Sharpe ratio beats threshold.

    sr is the monthly sample Sharpe ratio, skew the sample skewness
    and kurt the raw kurtosis, which is 3 for normal returns.
    """
    spread = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return ndtr((sr - threshold) * math.sqrt(months - 1) / spread)


def expected_max_sharpe(trials, variance):
    """Expected best Sharpe ratio of independent trials with no
    skill, whose sample Sharpe ratios have this variance."""
    if trials <= 1:
        return 0.0
    g = EULER_GAMMA
    first = (1 - g) * ndtri(1 - 1 / trials)
    second = g * ndtri(1 - 1 / (trials * math.e))
    return math.sqrt(variance) * (first + second)


def deflated_sharpe(returns, trials, variance):
    """PSR of one strategy against the best of `trials` lucky ones."""
    sr = returns.mean() / returns.std(ddof=1)
    skew, kurt = moments(returns)
    sr0 = expected_max_sharpe(trials, variance)
    return psr(sr, sr0, len(returns), skew, kurt)


def effective_trials(x):
    """(sum of eigenvalues)**2 / sum of squared eigenvalues of the
    candidates' return correlations, written without eigenvalues."""
    corr = np.corrcoef(x)
    return len(corr) ** 2 / (corr**2).sum()


@dataclass(frozen=True)
class Params:
    # the market
    stocks: int = 500
    months: int = 240  # 20 years of monthly returns
    styles: int = 10
    idio_vol: float = 0.10  # a stock's own monthly volatility
    style_vol: float = 0.005  # monthly volatility of a style return
    style_df: int = 5  # Student t degrees of freedom for styles
    true_sharpe: float = 0.7  # the true signal's annual Sharpe ratio

    # the candidates
    candidates: int = 200  # the true signal and 199 nulls
    shared: float = 0.5  # share of a null signal's variance from styles

    # the research
    in_sample: int = 120  # months used to choose
    first_year: int = 60  # walk-forward starts after five years
    step: int = 12  # and chooses again every year
    alpha: float = 0.025  # one-sided, the usual t of 1.96
    hurdle: float = 3.0  # the higher t hurdle for new factors
    level: float = 0.95  # the PSR or DSR a strategy must reach

    # the experiment
    worlds: int = 1000
    seed: int = 2026


BASE = Params()
KS = (1, 2, 5, 10, 20, 50, 100, 200)
TRIALS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
EULER_GAMMA = 0.5772156649015329
A_YEAR = math.sqrt(12)  # turns a monthly Sharpe ratio into a yearly one


def premium(p):
    """The true signal's monthly payoff per standard deviation, set
    so that its long-short portfolio has Sharpe p.true_sharpe."""
    vol = math.sqrt(p.idio_vol**2 + p.styles * p.style_vol**2)
    return p.true_sharpe / A_YEAR * vol / math.sqrt(p.stocks)


def moments(x):
    """Sample skewness and raw kurtosis of one return series."""
    z = (x - x.mean()) / x.std()
    return float((z**3).mean()), float((z**4).mean())


def world(p, seed):
    """Every candidate's monthly returns in one world."""
    rng = np.random.default_rng(seed)
    true, styles, returns = draw_market(p, rng)
    return candidate_returns(p, rng, true, styles, returns)


def discoveries(p, t, pvalues):
    """Which candidates each rule calls significant."""
    return {
        "t > 1.96": t > ndtri(1 - p.alpha),
        "t > 3.0": t > p.hurdle,
        "Bonferroni": bonferroni(pvalues, p.alpha),
        "Holm": holm(pvalues, p.alpha),
        "BH": benjamini_hochberg(pvalues, p.alpha),
    }


def best(past):
    """The candidate with the highest Sharpe ratio so far."""
    return int(np.argmax(sharpe_stats(past)[0]))


def best_if_deflated(p):
    """best, but only when its DSR over every candidate passes."""

    def choose(past):
        sr = sharpe_stats(past)[0]
        k = int(np.argmax(sr))
        dsr = deflated_sharpe(past[k], len(past), sr.var(ddof=1))
        return k if dsr > p.level else None

    return choose


def walk_forward_world(p, x):
    """Out-of-sample returns of each procedure, and whether best
    held the true signal each year with 200 and with 20 tried."""
    hindsight = best(x)
    procedures = {
        "best": best,
        "best if deflated": best_if_deflated(p),
        "true signal": lambda past: 0,
        "chosen with hindsight": lambda past: hindsight,
    }

    out = {}
    for name, choose in procedures.items():
        returns, picks = walk_forward(p, x, choose)
        vol = returns.std(ddof=1)

        # a procedure that never trades has a Sharpe ratio of zero
        out[name] = A_YEAR * returns.mean() / vol if vol > 0 else 0.0
        if name == "best if deflated":
            traded = np.mean([k is not None for k in picks])

    held = {200: walk_forward(p, x, best)[1]}
    held[20] = walk_forward(p, x[:20], best)[1]
    held = {k: np.array(v) == 0 for k, v in held.items()}
    return out, held | {"traded": traded}


def deflation(p, x, sr):
    """PSR and DSR of the in-sample winner for many trial counts."""
    ins = x[:, : p.in_sample]
    win = int(np.argmax(sr))
    variance = sr.var(ddof=1)
    n_eff = effective_trials(ins)
    counts = TRIALS + (10, n_eff, p.candidates)
    scores = [deflated_sharpe(ins[win], n, variance) for n in counts]
    return np.array(scores), n_eff


def study(p, seed):
    """Everything the article measures in one world."""
    x = world(p, seed)
    sr, t, pvalues = sharpe_stats(x[:, : p.in_sample])
    held, _, _ = sharpe_stats(x[:, p.in_sample :])
    sr_all, t_all, p_all = sharpe_stats(x)

    rec = {"sr": sr, "held": held, "t": t}
    rec["winners"] = [int(np.argmax(sr[:k])) for k in KS]
    rec["rules"] = discoveries(p, t, pvalues)
    long_run = discoveries(p, t_all, p_all)
    rec["power_20y"] = {name: d[0] for name, d in long_run.items()}
    rec["wf"], rec["wf_held"] = walk_forward_world(p, x)
    rec["dsr"], rec["n_eff"] = deflation(p, x, sr)
    rec["true_20y"] = A_YEAR * sr_all[0]
    return rec


def run(p=BASE):
    """study on p.worlds independent worlds, one seed each."""
    seeds = np.random.SeedSequence(p.seed).spawn(p.worlds)
    return [study(p, s) for s in seeds]


def mean_se(x):
    """Mean and standard error of independent values."""
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def paired(a, b):
    """Mean and standard error of b minus a, world by world."""
    return mean_se(np.asarray(b) - np.asarray(a))


def stack(recs, key):
    return np.array([r[key] for r in recs])


def search_numbers(recs, p=BASE):
    """The winner's Sharpe ratios, in and out of sample, by K."""
    sr, held = stack(recs, "sr"), stack(recs, "held")
    winners = stack(recs, "winners")
    rows, rows_idx = [], np.arange(len(recs))
    null_var = sr[:, 1:].var(axis=1, ddof=1).mean()

    for j, k in enumerate(KS[1:], start=1):
        win = winners[:, j]
        best_null = sr[:, 1:k].max(axis=1).mean()
        sr0 = expected_max_sharpe(k - 1, null_var)

        rows.append(
            dict(
                k=k,
                true=float((win == 0).mean()),
                ins=A_YEAR * float(sr[rows_idx, win].mean()),
                out=A_YEAR * float(held[rows_idx, win].mean()),
                best_null=A_YEAR * float(best_null),
                sr0=A_YEAR * sr0,
            )
        )
    return rows, null_var


def winner_numbers(recs, p=BASE):
    """The 200-candidate winner against the true signal."""
    sr, held, t = (stack(recs, k) for k in ("sr", "held", "t"))
    win = stack(recs, "winners")[:, -1]
    idx = np.arange(len(recs))
    is_true = win == 0

    out = {"win_t": float(t[idx, win].mean())}
    out["true_t"] = float(t[:, 0].mean())
    out["true_in"] = A_YEAR * float(sr[:, 0].mean())
    out["true_out"] = A_YEAR * float(held[:, 0].mean())

    gap = paired(A_YEAR * held[idx, win], A_YEAR * sr[idx, win])
    out["win_drop"], out["win_drop_se"] = gap
    for tag, mask in (("null", ~is_true), ("true", is_true)):
        mean, se = mean_se(A_YEAR * held[idx, win][mask])
        out[f"win_{tag}_out"], out[f"win_{tag}_out_se"] = mean, se
    out["win_null_out_z"] = out["win_null_out"] / out["win_null_out_se"]

    # every null's hold-out Sharpe ratio must average zero
    nulls = A_YEAR * held[:, 1:].mean(axis=1)
    out["nulls_out"], out["nulls_out_se"] = mean_se(nulls)
    out["nulls_out_absz"] = abs(out["nulls_out"]) / out["nulls_out_se"]
    out["true_20y"], out["true_20y_se"] = mean_se(
        stack(recs, "true_20y")
    )
    return out


def rule_numbers(recs, p=BASE):
    """Discoveries, false ones, error rates and power by rule."""
    rows = {}
    for name in recs[0]["rules"]:
        found = np.array([r["rules"][name] for r in recs])
        false = found[:, 1:].sum(axis=1)
        total = found.sum(axis=1)

        rows[name] = dict(
            found=float(total.mean()),
            false=float(false.mean()),
            fwer=float((false > 0).mean()),
            fdp=float((false / np.maximum(total, 1)).mean()),
            power=float(found[:, 0].mean()),
            power_20y=float(
                np.mean([r["power_20y"][name] for r in recs])
            ),
        )

    # the Sharpe ratio each fixed hurdle asks for over ten years
    bonferroni_t = ndtri(1 - p.alpha / p.candidates)
    hurdles = {"t > 1.96": ndtri(1 - p.alpha), "t > 3.0": p.hurdle}
    hurdles["Bonferroni"] = bonferroni_t
    for name, t in hurdles.items():
        rows[name]["sharpe"] = A_YEAR * t / math.sqrt(p.in_sample)
    return rows, bonferroni_t


def walk_numbers(recs):
    """Out-of-sample Sharpe ratios and paired differences."""
    wf = {
        k: np.array([r["wf"][k] for r in recs]) for k in recs[0]["wf"]
    }
    out = {name: mean_se(v) for name, v in wf.items()}

    pairs = {
        "best_vs_true": ("true signal", "best"),
        "deflated_vs_true": ("true signal", "best if deflated"),
        "hindsight_vs_true": ("true signal", "chosen with hindsight"),
        "leak": ("best", "chosen with hindsight"),
    }
    diffs = {key: paired(wf[a], wf[b]) for key, (a, b) in pairs.items()}

    held = {}
    for key in (200, 20, "traded"):
        rows = np.array([r["wf_held"][key] for r in recs])
        held[key] = rows.mean(axis=0)
    return out, diffs, held


def dsr_numbers(recs, null_var, p=BASE):
    """Pass rates and precision of PSR and DSR for the winner."""
    scores = stack(recs, "dsr")
    passed = scores > p.level
    held = stack(recs, "held")
    win = stack(recs, "winners")[:, -1]
    is_true = win == 0
    win_out = A_YEAR * held[np.arange(len(recs)), win]

    named = {"psr": 0, "dsr_10": -3, "dsr_eff": -2, "dsr_200": -1}
    out = {}
    for tag, col in named.items():
        ok = passed[:, col]
        out[f"{tag}_pass"] = float(ok.mean())
        out[f"{tag}_true"] = (
            float(is_true[ok].mean()) if ok.any() else 0
        )
        out[f"{tag}_out"] = float(win_out[ok].mean()) if ok.any() else 0

    out["true_but_no"] = float((is_true & ~passed[:, -1]).mean())
    curves = {
        "null": passed[~is_true, : len(TRIALS)].mean(axis=0),
        "true": passed[is_true, : len(TRIALS)].mean(axis=0),
    }
    out["n_eff"] = float(stack(recs, "n_eff").mean())
    out["n_calibrated"] = calibrated_trials(recs, null_var, p)
    return out, curves


def calibrated_trials(recs, null_var, p=BASE):
    """N at which the expected maximum matches the simulated one."""
    target = stack(recs, "sr")[:, 1:].max(axis=1).mean()
    lo, hi = 1.0, 1e4
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if expected_max_sharpe(mid, null_var) < target:
            lo = mid
        else:
            hi = mid
    return mid


def example(recs, p=BASE):
    """The first world, in seed order, whose winner is a null."""
    for i, r in enumerate(recs):
        win = r["winners"][-1]
        if win != 0:
            return dict(
                world=i,
                ins=(A_YEAR * r["sr"]).tolist(),
                out=(A_YEAR * r["held"]).tolist(),
                winner=win,
                t=float(r["t"][win]),
            )


def table(rows, header, widths):
    """A fixed-width text table with every column right-aligned."""

    def line(cells):
        pairs = zip(cells, widths, strict=True)
        return "  ".join(str(c).rjust(w) for c, w in pairs)

    return "\n".join([line(header)] + [line(row) for row in rows])


def pct(x):
    return f"{100 * x:.0f}%"


def search_table(rows):
    cells = []
    for r in rows:
        nums = [
            f"{r[k]:.2f}" for k in ("ins", "out", "best_null", "sr0")
        ]
        cells.append([r["k"], pct(r["true"]), *nums])
    header = ["K", "winner true", "in-sample", "hold-out"]
    header += ["best null", "formula"]
    return table(cells, header, [3, 11, 9, 8, 9, 7])


def rule_table(rows):
    cells = []
    for name, r in rows.items():
        hurdle = f"{r['sharpe']:.2f}" if "sharpe" in r else "varies"
        nums = [f"{r['found']:.1f}", f"{r['false']:.1f}"]
        nums += [pct(r["fwer"]), pct(r["fdp"]), pct(r["power"])]
        cells.append([name, hurdle, *nums, pct(r["power_20y"])])
    header = ["rule", "Sharpe", "found", "false", "FWER", "FDP"]
    header += ["power", "20y power"]
    return table(cells, header, [10, 6, 5, 5, 4, 4, 5, 9])


def walk_table(out):
    cells = []
    for name, (mean, se) in out.items():
        cells.append([name, f"{mean:.2f}", f"{se:.2f}"])
    header = ["procedure", "Sharpe", "s.e."]
    return table(cells, header, [21, 6, 4])


def results(p=BASE):
    """Every number, table and chart series the article uses."""
    recs = run(p)
    search, null_var = search_numbers(recs, p)
    rules, bonferroni_t = rule_numbers(recs, p)
    walk, diffs, held = walk_numbers(recs)
    dsr, curves = dsr_numbers(recs, null_var, p)

    ex = example(recs, p)

    d = winner_numbers(recs, p) | dsr | model_numbers(p)
    d |= flatten(search, rules, walk, diffs)
    d |= example_numbers(ex, held)
    d["bonferroni_t"] = bonferroni_t
    percent(d)

    tables = {
        "search": search_table(search),
        "rules": rule_table(rules),
        "walk": walk_table(walk),
    }
    held_lists = {k: held[k].tolist() for k in (20, 200)}
    series = {"held": held_lists}
    series["curves"] = {k: v.tolist() for k, v in curves.items()}
    series["t_hist"] = t_histogram(stack(recs, "t"))
    series["example"], series["search"] = ex, search
    return {"derived": d, "tables": tables} | series


def model_numbers(p):
    """The IC, Grinold's estimate and Lo's standard error."""
    ic = premium(p) / math.sqrt(
        p.idio_vol**2 + p.styles * p.style_vol**2
    )

    out = {"ic": round(ic, 3), "grinold": ic * math.sqrt(12 * p.stocks)}
    out["lo_se"] = A_YEAR / math.sqrt(p.in_sample)  # at a Sharpe of 0
    return out


def flatten(search, rules, walk, diffs):
    """One name per number from the search, rules and walk-forward."""
    d = {}
    for r in search:
        for key in ("true", "ins", "out", "best_null", "sr0"):
            d[f"k{r['k']}_{key}"] = r[key]

    for name, r in rules.items():
        tag = name.replace(" > ", "_").replace(".", "")
        d |= {f"{tag}_{key}": v for key, v in r.items()}

    for name, (mean, se) in walk.items():
        d[f"wf_{name.replace(' ', '_')}"] = mean
        d[f"wf_{name.replace(' ', '_')}_se"] = se
    for key, (mean, se) in diffs.items():
        d[f"wf_{key}"], d[f"wf_{key}_se"] = mean, se

    d["wf_best_gap"] = -d["wf_best_vs_true"]
    d["wf_deflated_gap"] = -d["wf_deflated_vs_true"]
    return d


def example_numbers(ex, held):
    """The opening's world and the walk-forward shares."""
    w = ex["winner"]
    d = {"ex_in": ex["ins"][w], "ex_out": ex["out"][w], "ex_t": ex["t"]}
    d["held_200_first"], d["held_200_last"] = held[200][[0, -1]]
    d["held_20_first"], d["held_20_last"] = held[20][[0, -1]]
    d["deflated_traded"] = float(held["traded"])
    return d


def percent(d):
    """Every share among the numbers in d, written in percent."""
    shares = ("_true", "_fwer", "_fdp", "_power", "power_20y", "_pass")
    for key in d:
        ends = shares + ("_no", "_first", "_last", "_traded")
        if key.endswith(ends) and not key.startswith("wf_"):
            d[key] *= 100


def t_histogram(t):
    """Densities of the true signal's and the nulls' t-statistics."""
    bins = np.arange(-4, 6.01, 0.25)
    true = np.histogram(t[:, 0], bins, density=True)[0]
    nulls = np.histogram(t[:, 1:], bins, density=True)[0]
    out = {"bins": bins, "true": true, "null": nulls}
    return {key: v.tolist() for key, v in out.items()}


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


def round_x(ax, lo, hi):
    """Round, evenly spaced x ticks that cover lo to hi."""
    ticks = MaxNLocator(6, steps=[1, 2, 5, 10]).tick_values(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])


def chart_example(res, out):
    ex = res["example"]
    ins, held = np.array(ex["ins"]), np.array(ex["out"])
    w = ex["winner"]

    ax = plt.subplots()[1]
    dots = dict(s=8, lw=0, alpha=0.6, color=BLUE, label="nulls")
    ax.scatter(ins[1:], held[1:], **dots)
    ax.scatter(ins[0], held[0], s=40, color=ORANGE, label="true signal")
    ax.scatter(
        ins[w], held[w], s=40, color=GREEN, label="in-sample winner"
    )

    round_x(ax, ins.min(), ins.max())
    ax.set_xlabel("in-sample Sharpe ratio, first 10 years")
    finish(ax, "Hold-out Sharpe ratio, last 10 years", legend=True)
    save(ax.figure, out / "example.png")


def chart_search(res, out):
    rows = res["search"]
    ks = [r["k"] for r in rows]

    ax = plt.subplots()[1]
    ax.plot(
        ks,
        [r["ins"] for r in rows],
        color=BLUE,
        marker="o",
        label="in-sample",
    )

    ax.plot(
        ks,
        [r["out"] for r in rows],
        color=ORANGE,
        marker="o",
        label="hold-out",
    )
    ax.axhline(BASE.true_sharpe, color=GREY, lw=1, ls="--")

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(1.6, 250)
    ax.set_xlabel("candidates tried, on a log scale")
    finish(ax, "The winner's Sharpe ratio, a year", legend=True)
    save(ax.figure, out / "search.png")


def chart_t(res, out, p=BASE):
    hist = res["t_hist"]
    bins = np.array(hist["bins"])

    ax = plt.subplots()[1]
    for key, colour, label in (
        ("null", BLUE, "nulls"),
        ("true", ORANGE, "true signal"),
    ):
        dens = np.array(hist[key])
        steps = np.append(dens, dens[-1])  # the last bin's top
        ax.step(bins, steps, where="post", color=colour, label=label)

    hurdles = (1.96, p.hurdle, res["derived"]["bonferroni_t"])
    for x in hurdles:
        ax.axvline(x, color=GREY, lw=0.8, ls="--")

    ax.set_xticks(np.arange(-4, 6.01, 2))
    ax.set_xlim(-4, 6)
    ax.set_xlabel("in-sample t-statistic over 10 years")
    finish(ax, "Density of in-sample t-statistics", legend=True)
    save(ax.figure, out / "tstats.png")


def chart_held(res, out, p=BASE):
    held = res["held"]
    years = np.arange(len(held["200"])) + p.first_year // 12
    ax = plt.subplots()[1]
    for key, colour in (("200", BLUE), ("20", ORANGE)):
        share = 100 * np.array(held[key])
        ax.plot(
            years, share, color=colour, marker="o", label=f"{key} tried"
        )

    ax.set_xticks(range(5, 20, 2))
    ax.set_xlim(4.5, 19.5)
    ax.set_xlabel("years of history at the choice")
    finish(
        ax,
        "Worlds in which the yearly winner is the true signal, in %",
        legend=True,
    )
    save(ax.figure, out / "held.png")


def chart_dsr(res, out):
    curves = res["curves"]
    ax = plt.subplots()[1]
    for key, colour, label in (
        ("null", BLUE, "winner is a null"),
        ("true", ORANGE, "winner is the true signal"),
    ):
        share = 100 * np.array(curves[key])
        ax.plot(TRIALS, share, color=colour, marker="o", label=label)

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_xlim(0.8, 1250)
    ax.set_xlabel("trials counted in the DSR, on a log scale")
    finish(ax, "Winners that pass at 0.95, in %", legend=True)
    save(ax.figure, out / "dsr.png")


CHARTS = (chart_example, chart_search, chart_t, chart_held, chart_dsr)


def main(out=Path(".")):
    res = results()
    res = json.loads(json.dumps(res, default=float))
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for chart in CHARTS:
        chart(res, figures)

    text = json.dumps(res, indent=1) + "\n"
    (out / "results.json").write_text(text)
    print(*res["tables"].values(), sep="\n\n")


if __name__ == "__main__":
    main()
