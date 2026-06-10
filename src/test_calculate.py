"""Tests documenting the corrected matchpoint / IMP scoring in calculate.py.

Background
----------
``calculate_mp_score`` and ``calculate_imp_score`` rank candidate cards by
comparing each card against every other card across the simulated worlds
(playouts). The OLD ``calculate_mp_score`` was broken: it

  (a) ignored the per-world sample weights,
  (b) truncated every card's list to the shortest one (``check_array_lengths``),
  (c) compared lists that had been sorted INDEPENDENTLY per card.

(a)+(b)+(c) silently corrupt the result for any card with a wide / rare trick
spread (e.g. cashing an ace that usually does nothing but occasionally sets the
contract by a lot). In one real position this inflated the ace of diamonds to
~92/100 matchpoints. IMP scoring was less affected only because its magnitude
weighting washes the artifact out, whereas matchpoints are sign-only and
amplify it.

The corrected method
--------------------
Both functions now share ``_round_robin_score``: a PAIRED, per-world,
weight-aware round-robin -- identical to Brill's
``DdsResultAnalyzer.CalculateImpScores`` / ``EndgamePlayEngine`` (per-world
``si[w] - sj[w]`` weighted by ``worldWeights[w]``). Only the per-comparison
value and therefore the output scale differ:

    matchpoints:  value = sign(diff)          ->  result scale  -1 .. +1
    IMPs:         value = signed diff_to_imps  ->  result scale -24 .. +24

Contract: all score lists must be the SAME length and aligned by world index
(world ``k`` is the same deal for every card); independent sorting / histogram
aggregation is forbidden and now raises instead of silently truncating.

Brill parity
------------
The vectors and expected values below are deliberately simple integers so the
exact same cases can be dropped into a Brill xUnit test. Brill already has the
IMP round-robin (``CalculateImpScores`` / ``CalculateWeightedImpScores``); it
does NOT yet have the matchpoint counterpart -- it needs the same method with
``sign(diff)`` in place of ``DifferenceToImps`` and a -1..+1 output scale.

Run:  pytest src/test_calculate.py        (or)  python src/test_calculate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calculate  # noqa: E402


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Legacy implementation kept here ONLY to document the bug being fixed.
# This is the old calculate_mp_score verbatim (0..100 scale, min-truncation,
# weight-ignoring). Do not use it in production.
# ---------------------------------------------------------------------------
def _legacy_mp_score(data):
    scores = {key: 0 for key in data}
    keys = list(data.keys())
    num_arrays = len(keys)
    num_plays = min(len(v) for v in data.values())  # check_array_lengths
    if num_arrays == 1:
        scores[keys[0]] = 100
        return scores
    data_lists = {k: [float(v) for v in vals[:num_plays]] for k, vals in data.items()}
    for i in range(num_arrays):
        for j in range(num_arrays):
            if i != j:
                for k in range(num_plays):
                    if data_lists[keys[i]][k] > data_lists[keys[j]][k]:
                        scores[keys[i]] += 1
                    elif data_lists[keys[i]][k] < data_lists[keys[j]][k]:
                        scores[keys[i]] -= 1
    max_mp_score = 2 * num_plays * (num_arrays - 1)
    for key in scores:
        scores[key] = round(100 * (scores[key] + max_mp_score / 2) / max_mp_score)
    return scores


# ===========================================================================
# Matchpoint scoring  (scale -1 .. +1)
# ===========================================================================
def test_mp_two_cards_unweighted():
    # A beats B on worlds 0,1 (+1 each) and loses on world 2 (-1).
    # acc_A = +1+1-1 = 1 ; norm = W*(n-1) = 3*1 = 3 ; A = 1/3.
    data = {"A": [100, 100, -50], "B": [50, 50, 50]}
    s = calculate.calculate_mp_score(data)
    assert approx(s["A"], 1 / 3), s
    assert approx(s["B"], -1 / 3), s


def test_mp_equal_weights_are_a_no_op():
    # Weighting every world by the same constant must not change anything.
    data = {"A": [100, 100, -50], "B": [50, 50, 50]}
    assert calculate.calculate_mp_score(data) == \
        calculate.calculate_mp_score(data, weights=[2, 2, 2])


def test_mp_weighting_can_flip_the_winner():
    # Same outcomes, but the world A LOSES (world 2) is now 4x as likely.
    # acc_A = +1 +1 -4 = -2 ; W = 6 ; A = -2/6 = -1/3  -> A now worse than B.
    data = {"A": [100, 100, -50], "B": [50, 50, 50]}
    s = calculate.calculate_mp_score(data, weights=[1, 1, 4])
    assert approx(s["A"], -1 / 3), s
    assert approx(s["B"], 1 / 3), s


def test_mp_tie_counts_as_zero():
    # world 0 is a tie (0), world 1 A wins (+1) -> acc_A = 1 ; norm = 2 ; A = 0.5
    data = {"A": [100, 100], "B": [100, 50]}
    s = calculate.calculate_mp_score(data)
    assert approx(s["A"], 0.5), s
    assert approx(s["B"], -0.5), s


def test_mp_strict_domination_is_plus_one():
    data = {"X": [100, 100, 100], "Y": [50, 50, 50]}
    s = calculate.calculate_mp_score(data)
    assert approx(s["X"], 1.0) and approx(s["Y"], -1.0), s


def test_mp_single_card_is_neutral_zero():
    # Changed from the old behaviour (which returned 100): with no field to
    # compare against the score is neutral 0 on both scales.
    assert calculate.calculate_mp_score({"A": [100, 50]}) == {"A": 0.0}


# ===========================================================================
# IMP scoring  (scale -24 .. +24), SAME method as MP
# ===========================================================================
def test_imp_two_cards_unweighted():
    # diff 230 -> +6 IMP (worlds 0,1); diff -270 -> -7 IMP (world 2).
    # acc_A = 6+6-7 = 5 ; norm = 3 ; A = 5/3 = 1.6667
    data = {"A": [400, 400, -100], "B": [170, 170, 170]}
    s = calculate.calculate_imp_score(data)
    assert approx(s["A"], 5 / 3, tol=1e-2), s
    assert approx(s["B"], -5 / 3, tol=1e-2), s


def test_imp_weighting_can_flip_the_winner():
    # The world A loses (world 2, -7 IMP) is now twice as likely.
    # acc_A = 6 + 6 - 7*2 = -2 ; W = 4 ; A = -0.5
    data = {"A": [400, 400, -100], "B": [170, 170, 170]}
    s = calculate.calculate_imp_score(data, weights=[1, 1, 2])
    assert approx(s["A"], -0.5, tol=1e-2), s
    assert approx(s["B"], 0.5, tol=1e-2), s


def test_imp_equal_weights_are_a_no_op():
    data = {"A": [400, 400, -100], "B": [170, 170, 170]}
    assert calculate.calculate_imp_score(data) == \
        calculate.calculate_imp_score(data, weights=[3, 3, 3])


def test_imp_reflects_magnitude_where_mp_saturates():
    # Both rank X top, but IMP shows the size of the swing (diff 50 -> 2 IMP)
    # while MP saturates at +1.
    data = {"X": [100, 100, 100], "Y": [50, 50, 50]}
    assert approx(calculate.calculate_mp_score(data)["X"], 1.0)
    assert approx(calculate.calculate_imp_score(data)["X"], 2.0, tol=1e-2)


# ===========================================================================
# Input-contract enforcement (the heart of the fix)
# ===========================================================================
def test_unequal_lengths_raise_for_both():
    bad = {"A": [1, 2, 3], "B": [1, 2]}
    for fn in (calculate.calculate_mp_score, calculate.calculate_imp_score):
        try:
            fn(bad)
            assert False, f"{fn.__name__} should reject unequal-length lists"
        except ValueError:
            pass


def test_wrong_weight_length_raises():
    data = {"A": [1, 2, 3], "B": [3, 2, 1]}
    try:
        calculate.calculate_mp_score(data, weights=[1, 1])
        assert False, "should reject mismatched weights length"
    except ValueError:
        pass


# ===========================================================================
# Regression: the original "ace of diamonds = 92 MP" inflation
# ===========================================================================
def test_legacy_truncation_inflates_a_wide_spread_card():
    """Documents the OLD bug. With independently-sorted, unequal-length
    per-card histograms (best trick first), min-truncation keeps only the wide
    card's BEST entries and drops its common bad ones -> it is wrongly rated a
    top while a steady card is rated a bottom."""
    legacy = {
        "DA": [500, 300, 200, 100, 50, 10],  # wide spread, sorted best-first
        "S6": [120, 100],                     # steady, few distinct outcomes
    }
    s = _legacy_mp_score(legacy)
    assert s["DA"] == 100 and s["S6"] == 0, s  # clearly wrong


def test_new_method_rejects_the_corrupt_representation():
    """The corrected method refuses the malformed unequal-length input that the
    legacy code silently truncated."""
    corrupt = {"DA": [500, 300, 200, 100, 50, 10], "S6": [120, 100]}
    try:
        calculate.calculate_mp_score(corrupt)
        assert False, "should reject the unequal-length histogram representation"
    except ValueError:
        pass


def test_new_method_ranks_rare_upside_card_correctly():
    """Same situation expressed correctly: per-world aligned scores plus the
    world weights. DA is usually worthless (score 50) and only rarely sets the
    contract big (500), and those good worlds are LOW probability. The fixed,
    weighted method correctly rates DA far below the steady card."""
    data = {
        "DA": [50, 50, 50, 50, 50, 50, 50, 50, 500, 500],
        "S6": [120] * 10,
    }
    weights = [10, 10, 10, 10, 10, 10, 10, 10, 1, 1]  # good worlds are rare
    s = calculate.calculate_mp_score(data, weights=weights)
    # acc_DA = -1*10*8 + 1*1*2 = -78 ; W = 82 ; DA = -78/82 = -0.9512
    assert approx(s["DA"], -78 / 82), s
    assert approx(s["S6"], 78 / 82), s
    assert s["DA"] < s["S6"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
