"""Tests of the program extracted from the article (article.py).

Every random check uses seeds the article does not use (it uses 2026).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

import article as a

SMALL = replace(a.BASE, worlds=120, seed=7)


@pytest.fixture(scope="module")
def recs():
    return a.run(SMALL)


def test_multiple_testing_rules_on_known_lists():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # Bonferroni at 0.05 over 10 tests needs p <= 0.005
    assert a.bonferroni(p, 0.05).tolist() == [True] + [False] * 9
    # Holm: 0.001 <= 0.005, 0.008 > 0.05/9, so it stops after one
    assert a.holm(p, 0.05).sum() == 1
    # BH: the largest j with p_(j) <= 0.005 j is j = 2 (0.008 <= 0.010)
    assert a.benjamini_hochberg(p, 0.05).sum() == 2
    q = np.array([0.04, 0.001, 0.02, 0.03])
    # Holm: 0.001 <= 0.0125 but 0.02 > 0.05/3, so only one
    assert a.holm(q, 0.05).tolist() == [False, True, False, False]
    assert a.benjamini_hochberg(q, 0.05).tolist() == [True, True, True, True]


def test_rules_nest():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = rng.uniform(size=50) ** rng.uniform(1, 4)
        bon, holm, bh = (f(p, 0.05) for f in (a.bonferroni, a.holm, a.benjamini_hochberg))
        assert not (bon & ~holm).any()
        assert not (holm & ~bh).any()


def test_null_candidates_earn_zero(recs):
    held = np.array([r["held"] for r in recs])
    nulls = held[:, 1:].mean(axis=1)
    mean, se = a.mean_se(nulls)
    assert abs(mean) < 2.5 * se
    # and the in-sample winner, when it is a null, earns zero after
    win = np.array([r["winners"][-1] for r in recs])
    lucky = held[np.arange(len(recs)), win][win != 0]
    mean, se = a.mean_se(lucky)
    assert abs(mean) < 2.5 * se


def test_true_signal_has_the_planted_sharpe(recs):
    mean, se = a.mean_se([r["true_20y"] for r in recs])
    assert abs(mean - a.BASE.true_sharpe) < 2.5 * se


def test_nulls_sharing_a_style_are_correlated():
    p = SMALL
    rng = np.random.default_rng(5)
    true, styles, returns = a.draw_market(p, rng)
    style_ls = (styles * returns).mean(axis=1)
    own_sd = np.sqrt((returns**2).sum(axis=0)) / p.stocks
    own = own_sd * rng.standard_normal((2, p.months))
    mix = [(style_ls[0] + style_ls[k]) / math.sqrt(2) for k in (1, 2)]
    x = [math.sqrt(p.shared) * m + math.sqrt(1 - p.shared) * o for m, o in zip(mix, own)]
    corr = np.corrcoef(x)[0, 1]
    assert 0.1 < corr < 0.6


@pytest.mark.parametrize("trials", [10, 50, 200])
def test_expected_max_matches_simulation(trials):
    rng = np.random.default_rng(11)
    months = 120
    x = rng.standard_normal((1000, trials, months))
    sr = x.mean(axis=-1) / x.std(axis=-1, ddof=1)
    variance = sr.var(axis=1, ddof=1).mean()
    formula = a.expected_max_sharpe(trials, variance)
    assert abs(formula / sr.max(axis=1).mean() - 1) < 0.05


def test_psr_is_calibrated_for_single_null_strategies():
    rng = np.random.default_rng(13)
    passed = []
    for _ in range(4000):
        r = rng.standard_normal(120)
        sr = r.mean() / r.std(ddof=1)
        skew, kurt = a.moments(r)
        passed.append(a.psr(sr, 0.0, 120, skew, kurt) > 0.95)
    share = np.mean(passed)
    assert abs(share - 0.05) < 3 * math.sqrt(0.05 * 0.95 / 4000)


def test_moments_of_a_normal_sample():
    x = np.random.default_rng(17).standard_normal(200_000)
    skew, kurt = a.moments(x)
    assert abs(skew) < 0.02 and abs(kurt - 3) < 0.05


def test_effective_trials_counts_copies_once():
    rng = np.random.default_rng(19)
    base = rng.standard_normal((5, 2000))
    copies = np.repeat(base, 4, axis=0) + 1e-6 * rng.standard_normal((20, 2000))
    assert a.effective_trials(copies) == pytest.approx(5, rel=0.05)
    assert a.effective_trials(base) == pytest.approx(5, rel=0.05)


def test_walk_forward_uses_only_the_past():
    p = SMALL
    x = np.random.default_rng(23).standard_normal((3, p.months))
    seen = []

    def choose(past):
        seen.append(past.shape[1])
        return 1

    returns, held = a.walk_forward(p, x, choose)
    assert seen == list(range(p.first_year, p.months, p.step))
    assert len(returns) == p.months - p.first_year
    size = 1 / x[1, : p.first_year].std(ddof=1)
    assert returns[: p.step] == pytest.approx(size * x[1, p.first_year : p.first_year + p.step])


def test_more_candidates_inflate_the_winner_and_lower_its_hold_out(recs):
    rows, _ = a.search_numbers(recs, SMALL)
    first, last = rows[0], rows[-1]
    assert last["ins"] > first["ins"]
    assert last["out"] < first["out"]
    assert last["true"] < first["true"]


def test_deflation_with_all_candidates_rarely_passes_a_null(recs):
    scores = np.array([r["dsr"] for r in recs])
    win = np.array([r["winners"][-1] for r in recs])
    passed = scores[:, -1] > SMALL.level
    assert (passed & (win != 0)).mean() < 0.05
    # PSR against zero passes almost every winner
    assert (scores[:, 0] > SMALL.level).mean() > 0.9


def test_same_seed_same_world():
    s = np.random.SeedSequence(99)
    assert np.array_equal(a.world(SMALL, s), a.world(SMALL, s))
