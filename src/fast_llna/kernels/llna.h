/* llna.h: the single-source update kernel of fast_llna, shared by the C (CPU), CUDA and Metal hosts.
 *
 * Written in the common subset of C99, CUDA C++ and the Metal Shading Language: no standard headers,
 * no `long long`, no array parameters, no globals. Each host prepends:
 *   types     u32 (32-bit unsigned), llna_idx (64-bit signed), WORD (one or more u32 lanes of 32 replicas)
 *   macros    LLNA_FN (function qualifier), LLNA_PTR(T) (pointer to read-only buffer), MULHI(a, b),
 *             ZEROW (all-zero WORD), LOADW(ptr, off) (WORD at u32 offset off), ANYW(x) (any bit set),
 *             LLNA_LANE0(x) (lane 0 of a WORD as u32),
 *             RANDW(k0, k1, t, i, stream, w, d) (random digit word d for the lanes of WORD w)
 *   defines   PMAX (count bit planes, >= bitlen(max degree)), SEGMAX (>= max segments per degree),
 *             NCELL (partition cells), LLNA_D (random digits, 0 = deterministic), LLNA_CLAMP (0/1),
 *             LLNA_NP (noise period if it divides 32, else 32), LLNA_REPL (a 1 every LLNA_NP bits),
 *             LLNA_L (bits per node: 1..16 node-major fields with Wp == 1, or 32 for [N][Wp] words),
 *             LLNA_LMASK ((1 << LLNA_L) - 1), LLNA_COOP (1: coop > 1 lanes may share a node; needs the host
 *             macros LLNA_SHFL_XOR(x, m) (the value of x in lane ^ m), LLNA_BALLOT(b) (u32 mask of the
 *             lanes where b is true) and LLNA_CTZ(x) (index of the lowest set bit of x != 0))
 *
 * State layout: LLNA_L < 32: a flat u32 array in which node j's replicas are bits j * L .. j * L + L - 1
 * (Wp == 1, w == 0). LLNA_L == 32: u32 [N][Wp]; replica r is bit r % 32 of word r / 32. llna_update computes
 * the next state of node i for the replicas in WORD w (w is a u32 word offset).
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

/* The macros below expand inside llna_update and read its locals: Wp, w, s (own state), seg_cell, nseg, b0,
 * g[] (segment reach masks), bs (the single segment when L == 1) and d (the digit loop index, LLNA_OFFD). */
#define LLNA_SEL(m, a, b) ((b) ^ ((m) & ((a) ^ (b))))
#define LLNA_OFF1(s_, j_) ((llna_idx)((s_) * NCELL + (j_)) * Wp)
#define LLNA_OFFD(s_, j_) ((llna_idx)(((s_) * NCELL + (j_)) * LLNA_D + d) * Wp)

#if LLNA_L < 32 /* node j's replicas are bits j*L .. j*L + L - 1 of the flat word array (Wp == 1, w == 0) */
#define LLNA_GET(P, j)                                                                                 \
    ((LOADW(P, ((llna_idx)(j) * LLNA_L) >> 5) >> (u32)(((llna_idx)(j) * LLNA_L) & 31)) & LLNA_LMASK)
#else
#define LLNA_GET(P, j) LOADW(P, (llna_idx)(j) * Wp + w)
#endif

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

#if LLNA_L == 1 /* one replica: acc = the TBL row of the single segment bs the count falls in */
#define LLNA_PICK(acc, TBL, OFF)                                                                       \
    acc = LLNA_SEL(s, LOADW(TBL, OFF(1, seg_cell[2 * bs + 1]) + w), LOADW(TBL, OFF(0, seg_cell[2 * bs]) + w));
#else
#define LLNA_PICK(acc, TBL, OFF) LLNA_SELECT(acc, TBL, OFF)
#endif

LLNA_FN WORD llna_update(LLNA_PTR(u32) S, LLNA_PTR(int) indptr, LLNA_PTR(int) indices,
                         LLNA_PTR(int) seg_off, LLNA_PTR(int) seg_thr, LLNA_PTR(int) seg_cell,
                         LLNA_PTR(u32) ONE, LLNA_PTR(u32) HASF, LLNA_PTR(u32) FRAC, LLNA_PTR(u32) STREAM,
                         LLNA_PTR(u32) CM, LLNA_PTR(u32) CV, llna_idx Wp, u32 k0, u32 k1, u32 t, int i,
                         llna_idx w, int lane, int coop) {
    /* count of living in-neighbours e = lane, lane + coop, ... (coop lanes share node i, then sum) */
    int lo = indptr[i], k = indptr[i + 1] - lo, b0 = seg_off[k], nseg = seg_off[k + 1] - b0;
#if LLNA_L == 1 /* one replica: an integer count and threshold scan are cheaper than bit planes */
    int q = 0;
    for (int e = lane; e < k; e += coop) q += (int)LLNA_LANE0(LLNA_GET(S, indices[lo + e]));
#if LLNA_COOP
    for (int sh = 1; sh < coop; sh <<= 1) q += LLNA_SHFL_XOR(q, sh);
#endif
    int bs = 0; /* the count's segment: the last one whose first q is <= q */
    while (bs + 1 < nseg && seg_thr[b0 + bs + 1] <= q) bs++;
    bs += b0;
    WORD s = LLNA_GET(S, i);
#else
    /* bit-sliced: c[p] holds bit p of the count, per lane */
    int lim = 0, m = 0;
    WORD c[PMAX];
    for (int p = 0; p < PMAX; p++) c[p] = ZEROW;
    for (int e = lane; e < k; e += coop) {
        WORD x = LLNA_GET(S, indices[lo + e]);
        m++;
        if ((m & (m - 1)) == 0) lim++; /* lim = bitlen(m): m added counts still fit in lim planes */
        for (int p = 0; p < PMAX; p++) {
            if (p >= lim) break;
            WORD carry = c[p] & x;
            c[p] ^= x;
            x = carry;
        }
    }
#if LLNA_COOP
    if (coop > 1) { /* sum the coop lanes' partial counts: XOR butterfly of bit-sliced ripple adds */
        int kb = 0;
        while ((k >> kb) != 0) kb++; /* bitlen(k): every partial sum and the total fit in kb planes */
        for (int sh = 1; sh < coop; sh <<= 1) {
            WORD carry = ZEROW;
            for (int p = 0; p < PMAX; p++) {
                if (p >= kb) break;
                WORD a_ = c[p], y_ = LLNA_SHFL_XOR(a_, sh), ab_ = a_ ^ y_;
                c[p] = ab_ ^ carry;
                carry = (a_ & y_) | (carry & ab_);
            }
        }
        lim = kb;
    }
#endif
    WORD s = LLNA_GET(S, i);

    /* g[b]: lanes whose count reaches the first q of segment b (thresholds < 2^lim) */
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
#endif

    WORD out;
    LLNA_PICK(out, ONE, LLNA_OFF1)
#if LLNA_D > 0
    /* lanes with 0 < p < 1: alive iff U < p, comparing binary digits most significant first */
    WORD eq;
    LLNA_PICK(eq, HASF, LLNA_OFF1)
    if (ANYW(eq)) {
        WORD lt = ZEROW;
        for (int d = 0; d < LLNA_D; d++) {
            WORD pd;
            LLNA_PICK(pd, FRAC, LLNA_OFFD)
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
    out = (out & ~LLNA_GET(CM, i)) | LLNA_GET(CV, i);
#endif
    return out;
}

#if LLNA_COOP
/* GPU node kernel (Wp == 1, one lane per node): lane = i % 32 of a complete 32-lane SIMD group whose nodes are
 * i - lane .. i - lane + 31 (the grid is padded; lanes with i >= n only help). The group counts every node of
 * degree > hub together (coop = 32), each other lane updates its own node, then the 32 / L lanes that share an
 * output word OR their fields. Returns that word; the host stores it from lane % (32 / L) == 0 at word index
 * i * L / 32 when that is below the word count. */
LLNA_FN WORD llna_group(LLNA_PTR(u32) S, LLNA_PTR(int) indptr, LLNA_PTR(int) indices, LLNA_PTR(int) seg_off,
                        LLNA_PTR(int) seg_thr, LLNA_PTR(int) seg_cell, LLNA_PTR(u32) ONE, LLNA_PTR(u32) HASF,
                        LLNA_PTR(u32) FRAC, LLNA_PTR(u32) STREAM, LLNA_PTR(u32) CM, LLNA_PTR(u32) CV, u32 k0,
                        u32 k1, u32 t, int n, int hub, int i, int lane) {
    int big = i < n && indptr[i + 1] - indptr[i] > hub;
    u32 hubs = LLNA_BALLOT(big);
    WORD out = ZEROW;
    while (hubs != 0u) { /* the whole group counts each hub; its own lane keeps the result */
        int h = LLNA_CTZ(hubs);
        hubs &= hubs - 1u;
        WORD o = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, ONE, HASF, FRAC, STREAM, CM, CV,
                             (llna_idx)1, k0, k1, t, i - lane + h, 0, lane, 32);
        if (lane == h) out = o;
    }
    if (i < n && !big)
        out = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, ONE, HASF, FRAC, STREAM, CM, CV,
                          (llna_idx)1, k0, k1, t, i, 0, 0, 1);
    WORD v = out << (u32)(((llna_idx)i * LLNA_L) & 31);
    for (int sh = 1; sh < 32 / LLNA_L; sh <<= 1) v |= LLNA_SHFL_XOR(v, sh);
    return v;
}
#endif
