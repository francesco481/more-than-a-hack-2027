"""Shared Viterbi possession & dead-ball decoders (used by both Metrica CV and inference).

All tunable decoder hyperparameters live in `models/decoder_params.json`, calibrated
strictly via Leave-One-Game-Out cross-validation on public Metrica Sample Games 1-3.
"""

import numpy as np


def logit_from_prob(prob, eps=1e-4):
    """Convert probabilities in [0, 1] to clamped log-odds log(p / (1 - p))."""
    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


def runs_of_mask(mask):
    """Return [(start, stop)] half-open intervals where boolean `mask` is True."""
    m = np.asarray(mask, dtype=bool)
    if len(m) == 0:
        return []
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def viterbi_binary(emission_diff, switch_penalty, init_bias=0.0, boundary_prob=None, boundary_alpha=0.0):
    """Exact 2-state Viterbi decoder over states {0, 1}.

    Parameters
    ----------
    emission_diff : array-like, shape (T,)
        Signed log-odds log P(state=1) / P(state=0) per frame.
    switch_penalty : float
        Base log-penalty paid for switching between state 0 and state 1.
    init_bias : float
        Log-prior at frame 0 favoring state 1 (>0) or state 0 (<0).
    boundary_prob : array-like or None
        Optional per-frame transition probability in [0, 1] that lowers switch cost.
    boundary_alpha : float
        Strength of boundary modulation: cost(t) = switch_penalty * max(0.25, 1 - boundary_alpha * p(t)).
    """
    diff = np.asarray(emission_diff, dtype=np.float32)
    n = len(diff)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    half_e = 0.5 * diff
    if boundary_prob is not None and boundary_alpha > 0.0:
        pb = np.clip(np.asarray(boundary_prob, dtype=np.float32), 0.0, 1.0)
        pen = float(switch_penalty) * np.maximum(0.25, 1.0 - float(boundary_alpha) * pb)
    else:
        pen = np.full(n, float(switch_penalty), dtype=np.float32)

    dp0 = np.empty(n, dtype=np.float64)
    dp1 = np.empty(n, dtype=np.float64)
    back0 = np.empty(n, dtype=np.int8)
    back1 = np.empty(n, dtype=np.int8)

    dp0[0] = -half_e[0] - 0.5 * init_bias
    dp1[0] = half_e[0] + 0.5 * init_bias
    back0[0] = 0
    back1[0] = 1

    for t in range(1, n):
        c = float(pen[t])
        e = float(half_e[t])

        stay0 = dp0[t - 1]
        from1 = dp1[t - 1] - c
        if stay0 >= from1:
            dp0[t] = stay0 - e
            back0[t] = 0
        else:
            dp0[t] = from1 - e
            back0[t] = 1

        stay1 = dp1[t - 1]
        from0 = dp0[t - 1] - c
        if stay1 >= from0:
            dp1[t] = stay1 + e
            back1[t] = 1
        else:
            dp1[t] = from0 + e
            back1[t] = 0

    states = np.empty(n, dtype=np.int8)
    states[-1] = 1 if dp1[-1] >= dp0[-1] else 0
    for t in range(n - 1, 0, -1):
        states[t - 1] = back1[t] if states[t] == 1 else back0[t]
    return states


def absorb_short_runs(
    states, min_len, protected_mask=None,
    boundary_prob=None, short_min_len=None, boundary_thresh=0.50, boundary_radius=3,
    emission_diff=None, logit_thresh=None, dead_prob=None, suffix_dead_thr=1.0,
):
    """Iteratively merge contiguous runs shorter than `min_len` into an adjacent run."""
    out = np.asarray(states).copy()
    n = len(out)
    if n == 0 or min_len <= 1:
        return out

    pb = np.asarray(boundary_prob, dtype=np.float32) if boundary_prob is not None else None
    ed = np.asarray(emission_diff, dtype=np.float32) if emission_diff is not None else None
    dp = np.asarray(dead_prob, dtype=np.float32) if dead_prob is not None else None
    has_short = short_min_len is not None and int(short_min_len) < int(min_len)
    use_bg = has_short and pb is not None
    use_lg = has_short and ed is not None and logit_thresh is not None

    while True:
        breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [n]))
        n_runs = len(bounds) - 1
        if n_runs <= 1:
            break

        lengths = bounds[1:] - bounds[:-1]
        candidates = []
        for i in range(n_runs):
            a, b = int(bounds[i]), int(bounds[i + 1])
            if lengths[i] < min_len:
                if protected_mask is not None and np.any(protected_mask[a:b]):
                    continue
                if has_short and lengths[i] >= int(short_min_len):
                    if use_lg:
                        if a == 0 or b == n:
                            sign = 1.0 if out[a] == 1 else -1.0
                            not_dead_tail = (b < n) or (dp is None) or (float(np.mean(dp[a:b])) < float(suffix_dead_thr))
                            if not_dead_tail and float(np.mean(sign * ed[a:b])) >= float(logit_thresh):
                                continue
                    elif use_bg and a > 0 and b < n:
                        r = int(boundary_radius)
                        p_start = float(np.max(pb[max(0, a - r):min(n, a + r + 1)]))
                        p_end = float(np.max(pb[max(0, b - r):min(n, b + r + 1)]))
                        if p_start >= float(boundary_thresh) and p_end >= float(boundary_thresh):
                            continue
                candidates.append(i)
        if not candidates:
            break

        idx = min(candidates, key=lambda i: lengths[i])
        a, b = int(bounds[idx]), int(bounds[idx + 1])
        out[a:b] = out[a - 1] if a > 0 else out[b]

    return out


def semi_markov_domino_split(
    states, local_prob, min_keep=50, min_cand=75, prob_thresh=0.52, mean_thresh=0.72,
):
    """Semi-Markov turnover recovery inside long continuous possessions.

    When a global sequence model swallows a medium-duration turnover (>= `min_cand` frames)
    between two same-team spells (causing a 3x domino merge T1 -> ~T -> T2), split [ca, cb)
    if the local frame-level expert (`local_prob` = P(Home|live)) sustains opposite-team
    probability >= `prob_thresh` (13-frame smoothed) with segment mean >= `mean_thresh`,
    while preserving >= `min_keep` frames on any non-empty flanking piece.
    """
    out = np.asarray(states, dtype=np.int8).copy()
    n = len(out)
    if n < min_keep + min_cand or local_prob is None:
        return out

    ph = np.asarray(local_prob, dtype=np.float64)
    breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n]))

    for idx in range(len(bounds) - 1):
        r0, r1 = int(bounds[idx]), int(bounds[idx + 1])
        if r1 - r0 < min_keep + min_cand:
            continue
        val = int(out[r0])
        p_opp_raw = (1.0 - ph[r0:r1]) if val == 1 else ph[r0:r1]
        m = len(p_opp_raw)
        cs = np.concatenate(([0.0], np.cumsum(p_opp_raw)))

        in_span = False
        s_a = 0
        raw_spans = []
        for t in range(m):
            lo_w = max(0, t - 6)
            hi_w = min(m, t + 7)
            p_sm = (cs[hi_w] - cs[lo_w]) / (hi_w - lo_w)
            if p_sm >= prob_thresh and not in_span:
                in_span = True
                s_a = t
            elif p_sm < prob_thresh and in_span:
                in_span = False
                raw_spans.append([r0 + s_a, r0 + t])
        if in_span:
            raw_spans.append([r0 + s_a, r1])

        merged = []
        for a, b in raw_spans:
            if merged and a - merged[-1][1] <= 15:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        cands = [(a, b) for a, b in merged if b - a >= min_cand]

        cur_left = r0
        for ca, cb in cands:
            if ca - cur_left < 15:
                ca = cur_left
            elif ca - cur_left < min_keep:
                ca = cur_left + min_keep
            if r1 - cb < 15:
                cb = r1
            elif r1 - cb < min_keep:
                cb = r1 - min_keep
            if cb - ca < min_cand:
                continue
            if (0 < ca - cur_left < min_keep) or (0 < r1 - cb < min_keep):
                continue
            if ca == cur_left and cb == r1:
                continue
            mean_p = float(np.mean((1.0 - ph[ca:cb]) if val == 1 else ph[ca:cb]))
            if mean_p >= mean_thresh:
                out[ca:cb] = 1 - val
                cur_left = cb

    return out


def refine_boundaries_local(
    states, local_logit, boundary_prob=None, max_shift=20, min_len=40,
):
    """ASRF-style local boundary refinement around internal H<->A transitions.

    Shifts each transition `t0` within `[t0 - max_shift, t0 + max_shift]` (subject to
    keeping both adjacent spells >= `min_len` frames) to maximise cumulative local
    log-odds + `3.0 * boundary_prob[t] - 0.03 * |t - t0|`.
    """
    out = np.asarray(states, dtype=np.int8).copy()
    n = len(out)
    if n == 0 or local_logit is None or max_shift <= 0:
        return out

    ll = np.asarray(local_logit, dtype=np.float64)
    pb = np.asarray(boundary_prob, dtype=np.float64) if boundary_prob is not None else None
    breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n])).tolist()
    if len(bounds) - 1 <= 1:
        return out

    for idx in range(len(bounds) - 2):
        r0 = bounds[idx]
        t0 = bounds[idx + 1]
        r1 = bounds[idx + 2]
        v0 = int(out[r0])
        v1 = int(out[t0])
        lo = max(r0 + int(min_len), t0 - int(max_shift))
        hi = min(r1 - int(min_len), t0 + int(max_shift))
        if lo >= hi:
            continue

        sign_right = 1.0 if v1 == 1 else -1.0
        best_t = t0
        best_score = -1e18
        cur_score = 0.0
        for t in range(lo, hi + 1):
            b_bonus = (3.0 * float(pb[t])) if (pb is not None and t < len(pb)) else 0.0
            score = cur_score + b_bonus - 0.03 * abs(t - t0)
            if score > best_score:
                best_score = score
                best_t = t
            if t < hi:
                cur_score += -sign_right * float(ll[t])

        if best_t != t0:
            if best_t < t0:
                out[best_t:t0] = v1
            else:
                out[t0:best_t] = v0
            bounds[idx + 1] = best_t

    return out
