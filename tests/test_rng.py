import numpy as np
import pytest

from fast_llna import rng

# Random123 kat_vectors, philox4x32 with 10 rounds: (ctr0..3, key0..1) -> out0..3
KAT = [
    ((0, 0, 0, 0, 0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 6, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize("inp, out", KAT)
def test_philox_known_answers(inp, out):
    got = rng.philox4x32(*(np.uint32(v) for v in inp))
    assert tuple(int(x) for x in got) == out


def test_philox_is_vectorised():
    c = np.arange(5, dtype=np.uint32)
    vec = rng.philox4x32(c, c, c, c, np.uint32(7), np.uint32(9))
    for n in range(5):
        one = rng.philox4x32(*(np.uint32(n),) * 4, np.uint32(7), np.uint32(9))
        assert all(int(v[n]) == int(o) for v, o in zip(vec, one, strict=True))


def test_rule_words_follow_the_contract():
    seed, t, i, w = 2**40 + 5, 3, 11, 2
    for d in range(9):
        expected = rng.philox4x32(
            np.uint32(t), np.uint32(i), np.uint32(w), np.uint32(d >> 2), np.uint32(5), np.uint32(2**8)
        )[d & 3]
        assert int(rng.rule_words(seed, t, i, w, d)) == int(expected)


@pytest.mark.parametrize(
    "p, P, depth",
    [(0.0, 0, 0), (1.0, 2**32, 0), (0.5, 2**31, 1), (0.25, 2**30, 2), (0.75, 3 * 2**30, 2), (2**-32, 1, 32)],
)
def test_quantize_exact_dyadics(p, P, depth):
    got_P, got_depth = rng.quantize(np.array([p]))
    assert int(got_P[0]) == P and got_depth == depth


def test_quantize_rounds_to_2_pow_minus_32_and_reports_bucketed_depth():
    P, depth = rng.quantize(np.array([0.3, 0.5]))
    assert int(P[0]) == round(0.3 * 2**32) and depth == 32
    assert rng.quantize(np.array([0.375]))[1] == 4  # exact depth 3 -> bucket 4


def test_quantize_warns_when_probability_vanishes():
    with pytest.warns(UserWarning):
        rng.quantize(np.array([1e-12]))
