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


def test_rules_reject_nan_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fl.Rules(fl.symmetric(3), np.full((1, 2, 3), np.nan))


def test_rules_keep_a_private_read_only_copy_of_p():
    p = np.zeros((1, 2, 3))
    rules = fl.Rules(fl.symmetric(3), p)
    p[0, 0, 0] = 7.0
    assert rules.p[0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        rules.p[0, 0, 0] = 1.0


def test_take_accepts_a_scalar_index():
    rules = fl.life_like(5, [1, 2, 3], 7)
    assert np.array_equal(rules.take(1).p, rules.p[[1]])


def test_even_convention_is_irrelevant_for_odd_r():
    assert fl.symmetric(5) == fl.symmetric(5, "-+")
    assert fl.symmetric(4) != fl.symmetric(4, "-+")


def test_equivalent_validates_and_broadcasts_like_life_like():
    with pytest.raises(ValueError, match=r"\[0, 32\)"):
        fl.equivalent(5, 40, 0)
    b, s = fl.equivalent(5, 3, [1, 2])
    assert b.shape == s.shape == (2,)
    assert (b[0], s[0]) == fl.equivalent(5, 3, 1)


@pytest.mark.parametrize("r", range(1, 11))
def test_self_equivalent_rows_are_fixed_points_sorted_by_beta(r):
    codes = fl.self_equivalent(r)
    assert codes.shape == (2**r, 2) and codes.dtype == np.int64
    assert np.array_equal(codes[:, 0], np.arange(2**r))
    b, s = fl.equivalent(r, codes[:, 0], codes[:, 1])
    assert np.array_equal(b, codes[:, 0]) and np.array_equal(s, codes[:, 1])


@pytest.mark.parametrize("r", range(1, 5))
def test_self_equivalent_is_exactly_the_set_of_fixed_points(r):
    beta, sigma = np.meshgrid(np.arange(2**r), np.arange(2**r), indexing="ij")
    b, s = fl.equivalent(r, beta, sigma)
    brute = np.argwhere((b == beta) & (s == sigma))
    assert np.array_equal(brute, fl.self_equivalent(r))


@pytest.mark.parametrize("r", range(1, 11))
def test_quiescent_self_equivalent_rules_are_half(r):
    codes = fl.self_equivalent(r)
    assert len(codes[codes[:, 0] % 2 == 0]) == 2 ** (r - 1)


EXACT_DEGREES = (1, 2, 3, 7, 8, 12)


def _exact_cases():
    return [
        fl.life_like(5, [3, 17, 30], [9, 0, 21]),
        fl.life_like(6, [3, 17, 60], [9, 0, 21], fl.symmetric(6, "+-")),
        fl.life_like(6, [3, 17, 60], [9, 0, 21], fl.symmetric(6, "-+")),
        fl.life_like(5, [3, 17, 30], [9, 0, 21], fl.uniform(5)),
        fl.majority(0.3),
    ]


@pytest.mark.parametrize("rules", _exact_cases())
def test_exact_copies_the_behaviour_per_degree_and_count(rules):
    t = rules.exact(EXACT_DEGREES)
    assert t.partition == fl.Partition("exact", sum(k + 1 for k in EXACT_DEGREES), degrees=EXACT_DEGREES)
    assert t.p.shape == (len(rules), 2, t.partition.ncell)
    off = 0
    for k in EXACT_DEGREES:
        for q in range(k + 1):
            for s in (0, 1):
                assert np.array_equal(t.p[:, s, off + q], rules.p[:, s, rules.partition.cell(q, k, s)])
        off += k + 1


def test_exact_partition_cells_and_validation():
    part = fl.Rules.exact(fl.majority(), (2, 5)).partition
    assert np.array_equal(part.cell([0, 2, 0, 5], [2, 2, 5, 5], 0), [0, 2, 3, 8])
    with pytest.raises(ValueError):
        part.cell(1, 3, 0)
    for bad in ((), (2, 1), (1, 1), (0, 2), (-1,)):
        with pytest.raises(ValueError):
            fl.Partition("exact", sum(k + 1 for k in bad), degrees=bad)
    with pytest.raises(ValueError):
        fl.Partition("exact", 4, degrees=(2,))
    with pytest.raises(ValueError):
        fl.Partition("uniform", 3, degrees=(2,))


def _distinct(tables):
    return len(np.unique(np.concatenate(tables).reshape(sum(len(t) for t in tables), -1), axis=0))


def _quiescent(r, even):
    codes = fl.self_equivalent(r)
    codes = codes[codes[:, 0] % 2 == 0]
    return fl.life_like(r, codes[:, 0], codes[:, 1], fl.symmetric(r, even)).exact((8,)).p


def test_quiescent_self_equivalent_rules_collapse_on_degree_8():
    assert _distinct([_quiescent(9, "+-")]) == 256
    assert _distinct([_quiescent(10, "+-"), _quiescent(10, "-+")]) == 256
    union = [_quiescent(r, e) for r in range(5, 11) for e in (("+-", "-+") if r % 2 == 0 else ("+-",))]
    assert _distinct(union) == 256


def test_rules_with_equal_tables_give_identical_trajectories():
    from _graphs import degree_graph

    graph = degree_graph([2, 4, 4, 6, 2, 4, 6, 8, 2, 4] * 3, seed=3)
    beta, sigma = 0b10011010, 0b01011010  # bits 3 and 4 agree: rho = 1/2 acts alike in both conventions
    init = fl.random_states(graph.n, 4, seed=1)
    runs = [
        fl.simulate(graph, fl.life_like(8, beta, sigma, fl.symmetric(8, e)), init, 6, backend="reference")
        for e in ("+-", "-+")
    ]
    exact = fl.life_like(8, beta, sigma, fl.symmetric(8, "+-")).exact((2, 4, 6, 8))
    runs.append(fl.simulate(graph, exact, init, 6, backend="reference"))
    assert np.array_equal(runs[0].states.bits, runs[1].states.bits)
    assert np.array_equal(runs[0].states.bits, runs[2].states.bits)
