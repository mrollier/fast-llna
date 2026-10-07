/* llna.h: the single-source update kernel of fast_llna, shared by the C (CPU), CUDA and Metal hosts.
 *
 * Written in the common subset of C99, CUDA C++ and the Metal Shading Language: no standard headers,
 * no `long long`, no array parameters, no globals. Each host prepends:
 *   types     u32 (32-bit unsigned), llna_idx (64-bit signed), WORD (one or more u32 lanes of 32 replicas)
 *   macros    LLNA_FN (function qualifier), LLNA_PTR(T) (pointer to read-only buffer), MULHI(a, b),
 *             ZEROW (all-zero WORD), LOADW(ptr, off) (WORD at u32 offset off), ANYW(x) (any bit set),
 *             RANDW(k0, k1, t, i, stream, w, d) (random digit word d for the lanes of WORD w)
 *   defines   PMAX (count bit planes, >= bitlen(max degree)), SEGMAX (>= max segments per degree),
 *             NCELL (partition cells), LLNA_D (random digits, 0 = deterministic), LLNA_CLAMP (0/1),
 *             LLNA_NP (noise period if it divides 32, else 32), LLNA_REPL (a 1 every LLNA_NP bits)
 *
 * State layout: u32 [N][Wp]; replica r is bit r % 32 of word r / 32. One call computes the next state of
 * node i for the replicas in WORD w (w is a u32 word offset).
 */

typedef struct {
    u32 x, y, z, w;
} llna_u32x4;

/* Philox4x32-10 (Salmon et al. 2011) */
LLNA_FN llna_u32x4 llna_philox(u32 c0, u32 c1, u32 c2, u32 c3, u32 k0, u32 k1) {
    for (int r = 0; r < 10; r++) {
        if (r) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        u32 hi0 = MULHI(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        u32 hi1 = MULHI(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        u32 n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    llna_u32x4 out;
    out.x = c0;
    out.y = c1;
    out.z = c2;
    out.w = c3;
    return out;
}

/* digit word d of the rule-output stream (purpose 0) at timestep t, node i, stream value sw (see rng.py) */
LLNA_FN u32 llna_rand(u32 k0, u32 k1, u32 t, u32 i, u32 sw, int d) {
    llna_u32x4 r = llna_philox(t, i, sw, (u32)(d >> 2), k0, k1);
    int m = d & 3;
    return m == 0 ? r.x : m == 1 ? r.y : m == 2 ? r.z : r.w;
}

#define LLNA_SEL(m, a, b) ((b) ^ ((m) & ((a) ^ (b))))
#define LLNA_OFF1(s_, j_) ((llna_idx)((s_) * NCELL + (j_)) * Wp)
#define LLNA_OFFD(s_, j_) ((llna_idx)(((s_) * NCELL + (j_)) * LLNA_D + d) * Wp)

/* acc = per lane, the TBL row of (own state, cell of the lane's count segment); telescoping over segments */
#define LLNA_SELECT(acc, TBL, OFF)                                                                     \
    {                                                                                                  \
        WORD prev_ = LLNA_SEL(s, LOADW(TBL, OFF(1, seg_cell[2 * b0 + 1]) + w),                         \
                              LOADW(TBL, OFF(0, seg_cell[2 * b0]) + w));                               \
        acc = prev_;                                                                                   \
        for (int b = 1; b < SEGMAX; b++) {                                                             \
            if (b >= nseg) break;                                                                      \
            WORD v_ = LLNA_SEL(s, LOADW(TBL, OFF(1, seg_cell[2 * (b0 + b) + 1]) + w),                  \
                               LOADW(TBL, OFF(0, seg_cell[2 * (b0 + b)]) + w));                        \
            acc ^= g[b] & (v_ ^ prev_);                                                                \
            prev_ = v_;                                                                                \
        }                                                                                              \
    }

LLNA_FN WORD llna_update(LLNA_PTR(u32) S, LLNA_PTR(int) indptr, LLNA_PTR(int) indices,
                         LLNA_PTR(int) seg_off, LLNA_PTR(int) seg_thr, LLNA_PTR(int) seg_cell,
                         LLNA_PTR(u32) ONE, LLNA_PTR(u32) HASF, LLNA_PTR(u32) FRAC, LLNA_PTR(u32) STREAM,
                         LLNA_PTR(u32) CM, LLNA_PTR(u32) CV, llna_idx Wp, u32 k0, u32 k1, u32 t, int i,
                         llna_idx w) {
    /* bit-sliced count of living in-neighbours: c[p] holds bit p of the count, per lane */
    int lo = indptr[i], k = indptr[i + 1] - lo, lim = 0;
    WORD c[PMAX];
    for (int p = 0; p < PMAX; p++) c[p] = ZEROW;
    for (int e = 0; e < k; e++) {
        WORD x = LOADW(S, (llna_idx)indices[lo + e] * Wp + w);
        if (((e + 1) & e) == 0) lim++; /* lim = bitlen(e + 1): the count still fits in lim planes */
        for (int p = 0; p < PMAX; p++) {
            if (p >= lim) break;
            WORD carry = c[p] & x;
            c[p] ^= x;
            x = carry;
        }
    }
    WORD s = LOADW(S, (llna_idx)i * Wp + w);

    /* g[b]: lanes whose count reaches the first q of segment b (thresholds < 2^lim) */
    int b0 = seg_off[k], nseg = seg_off[k + 1] - b0;
    WORD g[SEGMAX];
    for (int b = 1; b < SEGMAX; b++) {
        if (b >= nseg) break;
        int thr = seg_thr[b0 + b];
        WORD ge = ~ZEROW;
        for (int p = 0; p < PMAX; p++) {
            if (p >= lim) break;
            ge = ((thr >> p) & 1) ? (c[p] & ge) : (c[p] | ge);
        }
        g[b] = ge;
    }

    WORD out;
    LLNA_SELECT(out, ONE, LLNA_OFF1)
#if LLNA_D > 0
    /* lanes with 0 < p < 1: alive iff U < p, comparing binary digits most significant first */
    WORD eq;
    LLNA_SELECT(eq, HASF, LLNA_OFF1)
    if (ANYW(eq)) {
        WORD lt = ZEROW;
        for (int d = 0; d < LLNA_D; d++) {
            WORD pd;
            LLNA_SELECT(pd, FRAC, LLNA_OFFD)
            WORD u = RANDW(k0, k1, t, i, STREAM, w, d);
#if LLNA_NP < 32 /* noise period p divides 32: lane r reads digit lane r % p */
            u = (u & ((1u << LLNA_NP) - 1u)) * LLNA_REPL;
#endif
            lt |= eq & ~u & pd;
            eq &= ~(u ^ pd);
            if (!ANYW(eq)) break;
        }
        out |= lt;
    }
#endif
#if LLNA_CLAMP
    out = (out & ~LOADW(CM, (llna_idx)i * Wp + w)) | LOADW(CV, (llna_idx)i * Wp + w);
#endif
    return out;
}
