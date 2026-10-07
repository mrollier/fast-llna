from fractions import Fraction

import numpy as np
import pytest

import fast_llna as fl

DEGREES = [*range(1, 65), 97, 199, 1000]


def _in(rho, lo, hi, closed_lo, closed_hi):
    return (lo <= rho if closed_lo else lo < rho) and (rho <= hi if closed_hi else rho < hi)


def oracle_symmetric(rho, r, side):
    """Interval index from the thesis definitions (ch05 eq. r-intervals, app03 R+ / R-), in exact arithmetic.

    side is None for odd r, '+' for R+ and '-' for R- (even r).
    """
    h = r // 2
    for j in range(r):
        lo, hi = Fraction(j, r), Fraction(j + 1, r)
        if r % 2:
            c = (True, False) if j < h else (True, True) if j == h else (False, True)
        elif side == "+":
            c = (True, False) if j < h - 1 else (True, True) if j == h - 1 else (False, True)
        else:
            c = (True, False) if j < h else (True, True) if j == h else (False, True)
        if _in(rho, lo, hi, *c):
            return j
    raise AssertionError("not a partition")


def oracle_uniform(rho, r):
    for j in range(r):
        if _in(rho, Fraction(j, r), Fraction(j + 1, r), True, j == r - 1):
            return j
    raise AssertionError("not a partition")


def _cells(partition, s):
    return [
        (q, k, int(partition.cell(np.array([q]), np.array([k]), s)[0])) for k in DEGREES for q in range(k + 1)
    ]


@pytest.mark.parametrize("r", range(1, 26, 2))
def test_symmetric_odd_matches_thesis_intervals(r):
    for s in (0, 1):
        for q, k, j in _cells(fl.symmetric(r), s):
            assert j == oracle_symmetric(Fraction(q, k), r, None), (r, q, k, s)


@pytest.mark.parametrize("r", range(2, 26, 2))
@pytest.mark.parametrize("even", ["+-", "-+"])
def test_symmetric_even_uses_r_plus_and_r_minus(r, even):
    side = dict(zip((0, 1), even, strict=True))  # '+-': dead nodes (B) on R+, living nodes (S) on R-
    for s in (0, 1):
        for q, k, j in _cells(fl.symmetric(r, even=even), s):
            assert j == oracle_symmetric(Fraction(q, k), r, side[s]), (r, q, k, s)


@pytest.mark.parametrize("r", range(1, 26))
def test_uniform_matches_left_closed_intervals(r):
    for q, k, j in _cells(fl.uniform(r), 0):
        assert j == oracle_uniform(Fraction(q, k), r), (r, q, k)


def test_majority_cells_split_at_one_half():
    for q, k, j in _cells(fl.MAJORITY, 0):
        assert j == (0 if 2 * q < k else 1 if 2 * q == k else 2)


def test_cell_is_monotone_in_q():
    for p in (fl.symmetric(9), fl.symmetric(4), fl.uniform(5), fl.MAJORITY):
        for k in DEGREES:
            for s in (0, 1):
                assert np.all(np.diff(p.cell(np.arange(k + 1), np.full(k + 1, k), s)) >= 0)


@pytest.mark.parametrize(
    "rule, equiv",
    [((5, 6, 28), (24, 19)), ((5, 11, 19), (6, 5)), ((5, 6, 19), (6, 19))],
)
def test_equivalent_matches_thesis_examples(rule, equiv):
    assert fl.equivalent(*rule) == equiv


@pytest.mark.parametrize("r", [1, 2, 3, 5, 8, 9])
def test_equivalent_is_an_involution(r):
    beta, sigma = np.meshgrid(np.arange(2**r), np.arange(2**r), indexing="ij")
    b2, s2 = fl.equivalent(r, *fl.equivalent(r, beta.ravel(), sigma.ravel()))
    assert np.array_equal(b2, beta.ravel()) and np.array_equal(s2, sigma.ravel())


@pytest.mark.parametrize("r", range(1, 10))
def test_nonequivalent_count(r):
    rules = fl.nonequivalent(r)
    assert rules.shape == (2 ** (2 * r - 1) + 2 ** (r - 1), 2)
    b, s = fl.equivalent(r, rules[:, 0], rules[:, 1])
    # every kept rule is the lexicographic minimum of its class
    assert np.all((rules[:, 0] < b) | ((rules[:, 0] == b) & (rules[:, 1] <= s)))


def test_life_like_table_is_game_of_life():
    rules = fl.life_like(9, 8, 12)
    assert len(rules) == 1 and rules.partition == fl.symmetric(9)
    assert np.array_equal(np.flatnonzero(rules.p[0, 0]), [3])  # born with 3 alive of 8
    assert np.array_equal(np.flatnonzero(rules.p[0, 1]), [2, 3])  # survive with 2 or 3


def test_life_like_broadcasts_arrays_and_take_selects_rows():
    rules = fl.life_like(5, [1, 2, 3], 7)
    assert rules.p.shape == (3, 2, 5)
    sub = rules.take([2, 2, 0])
    assert np.array_equal(sub.p, rules.p[[2, 2, 0]])


def test_life_like_rejects_codes_out_of_range():
    with pytest.raises(ValueError):
        fl.life_like(5, 32, 0)


def test_majority_rule_is_a_fair_coin_at_ties():
    rules = fl.majority()
    assert rules.partition == fl.MAJORITY
    assert np.array_equal(rules.p, [[[0, 0.5, 1], [0, 0.5, 1]]])
