"""Numba kernels for the card-dealing loops in sample.py.

sample_cards_vec and shuffle_cards_bidding_info deal cards one round at a time: every
round each sample still missing cards draws one random number, picks a receiver from
its hcp/shape weights (distr_vec / distr_ak_vec) and takes the card if that hand has
room. In numpy every round is a dozen fancy-indexing passes over all samples; here a
round is one compiled loop.

The random numbers are still drawn in Python, one rng.random(n_active) per round,
exactly as distr_vec does, so the generator advances identically. The arithmetic is
the same IEEE double arithmetic in the same order as the numpy code (sums and cumsums
over 2-3 receivers are sequential in numpy too), so every receiver, and therefore every
sample, is bit-for-bit the same.

Numba is optional: AVAILABLE is False without it and sample.py keeps its numpy loops.
"""

import os

import numpy as np

try:
    if os.environ.get("BEN_NO_NUMBA"):
        raise ImportError("disabled by BEN_NO_NUMBA")
    from numba import njit
    AVAILABLE = True
except Exception:
    # Not only ImportError: a numba that is installed but broken (e.g. llvmlite's DLL
    # missing from a frozen build) must also fall back to the numpy loops
    AVAILABLE = False


def _jit(fn):
    """njit with an on-disk cache where possible. A PyInstaller exe has no source file
    for numba to cache beside, and njit(cache=True) raises there, so compile uncached."""
    try:
        return njit(cache=True, nogil=True)(fn)
    except Exception:
        return njit(nogil=True)(fn)


if AVAILABLE:

    @_jit
    def _pick(xp, n, r):
        # distr_vec for one sample: xp holds the weights already clipped at 0 plus 0.1
        s = 0.0
        for j in range(n):
            s += xp[j]
        d = s + 1e-10
        idx = 0
        c = 0.0
        for j in range(n - 1):
            c += xp[j] / d
            if r > c:
                idx += 1
        # The last cumulative probability is set to exactly 1.0
        if r > 1.0:
            idx += 1
        return idx

    @_jit
    def deal_round(rnd, order, js, r_shp, r_hcp, cards_received, hands, caps,
                   cards_in_suit, honors, update_hcp, hcp_dec, update_shp, shp_dec):
        """One dealing round over the samples with js[i] < order.shape[1].

        rnd: one uniform number per such sample, in sample order.
        order: (samples, cards) the card each sample deals at each step.
        honors: True picks receivers like distr_ak_vec (shape and hcp weights), False
        like distr_vec (shape weights).
        Returns how many samples still have cards to deal (the next round's rnd size)."""
        n_samples = js.shape[0]
        width = order.shape[1]
        n_recv = cards_received.shape[1]
        xp = np.empty(n_recv)
        hp = np.empty(n_recv)
        k = 0
        remaining = 0
        for i in range(n_samples):
            j = js[i]
            if j >= width:
                continue
            card = order[i, j]
            suit = card // cards_in_suit
            r = rnd[k]
            k += 1
            if honors:
                # distr_ak_vec: normalise both weights, multiply, normalise, then distr_vec
                sa = 0.0
                sb = 0.0
                for t in range(n_recv):
                    a = r_shp[i, t, suit]
                    if a < 0.0:
                        a = 0.0
                    a += 0.1
                    xp[t] = a
                    sa += a
                    b = r_hcp[i, t]
                    if b < 0.0:
                        b = 0.0
                    b += 0.1
                    hp[t] = b
                    sb += b
                sq = 0.0
                for t in range(n_recv):
                    q = (xp[t] / sa) * (hp[t] / sb)
                    xp[t] = q
                    sq += q
                for t in range(n_recv):
                    q = xp[t] / sq
                    if q < 0.0:
                        q = 0.0
                    xp[t] = q + 0.1
            else:
                for t in range(n_recv):
                    a = r_shp[i, t, suit]
                    if a < 0.0:
                        a = 0.0
                    xp[t] = a + 0.1
            recv = _pick(xp, n_recv, r)
            if cards_received[i, recv] < caps[recv]:
                cards_received[i, recv] += 1
                hands[i, recv, card] += 1
                if update_hcp:
                    r_hcp[i, recv] -= hcp_dec
                if update_shp:
                    r_shp[i, recv, suit] -= shp_dec
                j += 1
                js[i] = j
            if j < width:
                remaining += 1
        return remaining


def deal(rng, order, r_shp, r_hcp, cards_received, hands, caps, cards_in_suit, honors,
         update_hcp, hcp_dec, update_shp, shp_dec):
    """Deal every card in order (samples, cards); the arrays are updated in place.
    Draws rng.random(n) for the n samples still dealing at each round, like the numpy
    loops in sample.py."""
    n_samples, width = order.shape
    js = np.zeros(n_samples, dtype=np.int64)
    if width == 0:
        return
    order = np.ascontiguousarray(order, dtype=np.int64)
    caps = np.ascontiguousarray(caps, dtype=np.int64)
    active = n_samples
    while active > 0:
        rnd = rng.random(active)
        active = deal_round(rnd, order, js, r_shp, r_hcp, cards_received, hands, caps,
                            cards_in_suit, honors, update_hcp, float(hcp_dec), update_shp, float(shp_dec))


def usable(*arrays):
    """The kernels update these arrays in place and assume float64 weights and int64
    counts; anything else goes through the numpy code."""
    if not AVAILABLE:
        return False
    for a, dtype in arrays:
        if a.dtype != dtype or not a.flags.c_contiguous:
            return False
    return True
