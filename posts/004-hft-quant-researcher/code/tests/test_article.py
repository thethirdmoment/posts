"""Tests of the program extracted from the article (article.py)."""

from dataclasses import replace

import numpy as np
import pytest

import article as a

SHORT = replace(a.BASE, seconds=600.0)


def test_rates_have_the_queue_reactive_shapes():
    small, large = a.rates(a.BASE, 2), a.rates(a.BASE, 20)
    assert small[0] == large[0]  # limit orders do not depend on the queue
    assert large[1] > small[1]  # cancellations grow with the queue
    assert large[2] < small[2]  # market orders hit short queues harder
    assert a.rates(a.BASE, a.BASE.max_queue)[0] == 0.0


def test_next_event_picks_flows_in_proportion_to_their_rates():
    flows = a.rates(a.BASE, 5) + a.rates(a.BASE, 8)
    edges = np.cumsum(flows) / sum(flows)
    for k, edge in enumerate(edges):
        wait, side, kind = a.next_event(a.BASE, 5, 8, 1.0, edge - 1e-9)
        assert (side, kind) == divmod(k, 3)
        assert wait == pytest.approx(1 / sum(flows))


def test_queue_ran_out_moves_the_price_towards_the_empty_side():
    p = a.BASE
    assert a.queue_ran_out(p, a.BID, 0.99, 0.0, 3, (9, 11)) == (0, 3, None)
    assert a.queue_ran_out(p, a.BID, 0.0, 0.99, 3, (9, 11)) == (-1, 9, 3)
    assert a.queue_ran_out(p, a.ASK, 0.0, 0.99, 3, (9, 11)) == (1, 3, 11)
    assert a.queue_ran_out(p, a.ASK, 0.0, 0.0, 3, (9, 11)) == (1, 9, 11)


def test_groups_mirror_around_zero():
    imb = np.array([-1.0, -0.6, -0.59, -0.2, -0.1, 0.0, 0.1, 0.2, 0.59, 0.6, 1.0])
    g = a.group(imb)
    np.testing.assert_array_equal(g, -a.group(-imb))
    assert list(g) == [-2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2]
    assert a.imbalance(30, 3) == pytest.approx(27 / 33)


def test_session_is_deterministic_and_queues_stay_positive():
    x, y, z = (a.session(SHORT, s) for s in (5, 5, 6))
    np.testing.assert_array_equal(x["t"], y["t"])
    assert not np.array_equal(x["t"][:50], z["t"][:50])
    assert x["qb"].min() >= 1 and x["qa"].min() >= 1
    assert np.all(np.diff(x["t"]) > 0)
    # the price moves one tick at a time, and only when a queue ran out
    moved = x["move"] != 0
    assert set(np.abs(x["move"][moved])) == {1}
    np.testing.assert_array_equal(np.diff(x["bid"]), x["move"][1:])
    assert np.all(x["q_before"][moved] == 1)


def test_queue_step_fills_at_the_front_and_counts_down_behind():
    ev = {"move": [0, 0], "side": [0, a.BID], "kind": [0, a.MARKET], "u": [0, 0.5], "q_before": [0, 4]}
    assert a.queue_step(ev, 1, a.BID, 0) == ("fill", 0)
    assert a.queue_step(ev, 1, a.BID, 3) == ("wait", 2)
    assert a.queue_step(ev, 1, a.ASK, 3) == ("wait", 3)
    ev["kind"][1] = a.CANCEL  # u * q = 2: removes a lot ahead only if 3 are ahead
    assert a.queue_step(ev, 1, a.BID, 3) == ("wait", 2)
    assert a.queue_step(ev, 1, a.BID, 2) == ("wait", 2)
    ev["move"][1] = -1
    assert a.queue_step(ev, 1, a.BID, 5)[0] == "fill"
    assert a.queue_step(ev, 1, a.ASK, 5)[0] == "away"


def test_crossing_pays_the_offer_and_posting_rests_at_the_bid():
    log, ev = a.prepare(a.session(SHORT, 3))
    i = 100
    cross, rest, late = a.three_routes(SHORT, ev, i, 1)
    assert cross == ev["bid"][i] + 1 and late is not None
    assert rest is None or rest == ev["bid"][i]
    cross, rest, late = a.three_routes(SHORT, ev, i, -1)
    assert cross == ev["bid"][i]


def test_a_filter_that_never_triggers_changes_nothing():
    log, ev = a.prepare(a.session(SHORT, 4))
    plain = a.rest_on_one_side(ev, a.BID, None, 0.0)
    loose = a.rest_on_one_side(ev, a.BID, 1.0, 0.0)  # imbalance is never below -1
    assert plain == loose and len(plain) > 0


@pytest.fixture(scope="module")
def other_seeds():
    """Six sessions the article never uses (it uses seeds 0 to 39)."""
    runs = [a.run_session(a.BASE, seed) for seed in range(40, 46)]
    return a.summary(runs)


def test_the_articles_claims_hold_on_other_seeds(other_seeds):
    s, values = other_seeds
    assert np.all(np.diff(s["p_up"]) > 0)  # the chance of an up move rises across the groups
    assert s["p_up"][4] > 0.6 and s["alternation"] > 0.5
    # over all readings the mid barely drifts next to the strong signal's move
    assert np.all(np.abs(s["all_move"][:5]) < 0.1 * s["move"][4][:5])
    assert abs(s["p_up"][0] + s["p_up"][4] - 1) < 0.05  # the groups mirror
    assert np.max(np.abs(s["move"][[0, 1, 3, 4], :5])) < 0.5  # within ten seconds, never the half-spread
    assert values[:, 1].mean() > 0 > values[:, 0].mean()  # posting pays, crossing loses
    assert s["filter_pnl"][0] / s["filter_fills"][0] > s["none_pnl"][0] / s["none_fills"][0]
    assert s["filter_fills"][0] < s["none_fills"][0]
    assert s["gain"][3] < 0  # with a delay the filter costs
