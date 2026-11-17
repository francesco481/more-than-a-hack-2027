"""Align the Broadcast recording (164,488 frames) to the Tactical reference timeline (199,319 frames).

Both videos run at 25 fps, but the broadcast recording starts at a different pre-match point and
cuts/pauses differently across halftime. This script aligns every valid broadcast segment onto
the tactical frame index [0..199318] by matching player constellation centroids + ball positions,
and caches:
  * cache/aligned_broadcast_ball.npy   (199319, 3) float32 -- broadcast ball on tactical clock
  * cache/aligned_broadcast_valid.npy  (199319,)   int8    -- 1=live wide cam, 0=TV replay/cut, -1=unmapped
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]


def team_centroids_and_ball(players, ball, slot_team, valid_mask=None):
    """Compute per-frame (home_x, home_y, away_x, away_y, ball_x, ball_y) in metres."""
    n = len(ball)
    out = np.full((n, 6), np.nan, dtype=np.float32)

    home_slots = np.flatnonzero(slot_team == 0)
    away_slots = np.flatnonzero(slot_team == 1)

    chunk = 20000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        p = np.asarray(players[start:stop, :, :2], dtype=np.float32)
        b = np.asarray(ball[start:stop, :2], dtype=np.float32)

        hx = np.nanmean(p[:, home_slots, 0], axis=1) * PITCH_LENGTH
        hy = np.nanmean(p[:, home_slots, 1], axis=1) * PITCH_WIDTH
        ax = np.nanmean(p[:, away_slots, 0], axis=1) * PITCH_LENGTH
        ay = np.nanmean(p[:, away_slots, 1], axis=1) * PITCH_WIDTH
        bx = b[:, 0] * PITCH_LENGTH
        by = b[:, 1] * PITCH_WIDTH

        out[start:stop, 0] = hx
        out[start:stop, 1] = hy
        out[start:stop, 2] = ax
        out[start:stop, 3] = ay
        out[start:stop, 4] = bx
        out[start:stop, 5] = by

    if valid_mask is not None:
        out[~valid_mask] = np.nan
    return out


def find_global_offset(tac_feat, brd_feat, tac_lo, tac_hi, brd_lo, brd_hi, offset_min, offset_max):
    """Find integer offset delta = tac_frame - brd_frame minimizing constellation + ball L1 error."""
    # Subsample every 5 frames for fast coarse sweep, then refine at step 1
    b_idx = np.arange(brd_lo, brd_hi, 5)
    b_sub = brd_feat[b_idx]
    valid_b = np.isfinite(b_sub[:, 0]) & np.isfinite(b_sub[:, 2])
    b_idx = b_idx[valid_b]
    b_sub = b_sub[valid_b]

    best_delta = 0
    best_err = np.inf

    for delta in range(offset_min, offset_max + 1, 2):
        t_idx = b_idx + delta
        in_range = (t_idx >= tac_lo) & (t_idx < tac_hi)
        if np.sum(in_range) < 500:
            continue
        t_sub = tac_feat[t_idx[in_range]]
        b_s = b_sub[in_range]
        both = np.isfinite(t_sub[:, 0]) & np.isfinite(t_sub[:, 2])
        if np.sum(both) < 500:
            continue
        diff = np.abs(t_sub[both, :4] - b_s[both, :4]).mean()
        if diff < best_err:
            best_err = diff
            best_delta = delta

    # Fine 1-frame refinement around best_delta using both players and ball
    b_idx_full = np.arange(brd_lo, brd_hi)
    b_full = brd_feat[b_idx_full]
    valid_full = np.isfinite(b_full[:, 0]) & np.isfinite(b_full[:, 2])
    b_idx_full = b_idx_full[valid_full]
    b_full = b_full[valid_full]

    best_fine_err = np.inf
    best_fine_delta = best_delta
    for delta in range(best_delta - 25, best_delta + 26):
        t_idx = b_idx_full + delta
        in_range = (t_idx >= tac_lo) & (t_idx < tac_hi)
        t_s = tac_feat[t_idx[in_range]]
        b_s = b_full[in_range]
        both_p = np.isfinite(t_s[:, 0]) & np.isfinite(t_s[:, 2])
        p_err = np.abs(t_s[both_p, :4] - b_s[both_p, :4]).mean()

        both_b = np.isfinite(t_s[:, 4]) & np.isfinite(b_s[:, 4])
        b_err = np.hypot(t_s[both_b, 4] - b_s[both_b, 4], t_s[both_b, 5] - b_s[both_b, 5]).mean() if np.any(both_b) else 10.0

        score = p_err + 0.5 * b_err
        if score < best_fine_err:
            best_fine_err = score
            best_fine_delta = delta

    return best_fine_delta, best_fine_err


def refine_segment_offset(tac_feat, brd_feat, b_start, b_stop, base_delta, n_tac):
    """Allow local +/- 15 frame adjustment per contiguous broadcast_valid segment."""
    b_idx = np.arange(b_start, b_stop)
    b_seg = brd_feat[b_idx]
    valid = np.isfinite(b_seg[:, 0]) & np.isfinite(b_seg[:, 2])
    if np.sum(valid) < 15:
        return base_delta
    b_idx = b_idx[valid]
    b_seg = b_seg[valid]

    best_d = base_delta
    best_score = np.inf
    for d in range(base_delta - 15, base_delta + 16):
        t_idx = b_idx + d
        ok = (t_idx >= 0) & (t_idx < n_tac)
        if np.sum(ok) < 10:
            continue
        t_seg = tac_feat[t_idx[ok]]
        b_s = b_seg[ok]
        both_p = np.isfinite(t_seg[:, 0]) & np.isfinite(t_seg[:, 2])
        if np.sum(both_p) < 10:
            continue
        p_err = np.abs(t_seg[both_p, :4] - b_s[both_p, :4]).mean()
        both_b = np.isfinite(t_seg[:, 4]) & np.isfinite(b_s[:, 4])
        if np.sum(both_b) >= 5:
            b_err = np.median(np.hypot(t_seg[both_b, 4] - b_s[both_b, 4],
                                       t_seg[both_b, 5] - b_s[both_b, 5]))
        else:
            b_err = p_err
        # Small regularization toward base_delta so noisy short segments don't drift
        score = p_err + 0.5 * b_err + 0.02 * abs(d - base_delta)
        if score < best_score:
            best_score = score
            best_d = d
    return best_d


def runs_of(mask):
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1).tolist(), np.flatnonzero(diff == -1).tolist()))


def main():
    tac_players = np.load(os.path.join(CACHE_DIR, "tactical_players.npy"), mmap_mode="r")
    tac_ball = np.load(os.path.join(CACHE_DIR, "tactical_ball.npy"), mmap_mode="r")
    tac_slot = np.load(os.path.join(CACHE_DIR, "tactical_slot_team.npy"))

    brd_players = np.load(os.path.join(CACHE_DIR, "broadcast_players.npy"), mmap_mode="r")
    brd_ball = np.load(os.path.join(CACHE_DIR, "broadcast_ball.npy"), mmap_mode="r")
    brd_slot = np.load(os.path.join(CACHE_DIR, "broadcast_slot_team.npy"))
    brd_valid = np.load(os.path.join(CACHE_DIR, "broadcast_valid.npy"))

    n_tac = len(tac_ball)
    n_brd = len(brd_ball)
    print("Computing constellation & ball descriptors...")
    tac_feat = team_centroids_and_ball(tac_players, tac_ball, tac_slot)
    brd_feat = team_centroids_and_ball(brd_players, brd_ball, brd_slot, brd_valid)

    blocks = half_video_blocks(GAME)
    h1_start, h1_end = blocks[0][1], blocks[0][2]
    h2_start, h2_end = blocks[1][1], blocks[1][2]

    # Search for global offset per half between tactical and broadcast timelines
    delta1, err1 = find_global_offset(tac_feat, brd_feat, h1_start, h1_end + 1, 0, n_brd // 2, -10000, 30000)
    delta2, err2 = find_global_offset(tac_feat, brd_feat, h2_start, h2_end + 1, n_brd // 2, n_brd, 0, 50000)
    print("Global offsets (tactical_frame - broadcast_frame):")
    print("  1st Half: delta = {:+d} frames ({:+.2f} s), score = {:.2f} m".format(delta1, delta1 / 25.0, err1))
    print("  2nd Half: delta = {:+d} frames ({:+.2f} s), score = {:.2f} m".format(delta2, delta2 / 25.0, err2))

    # Map broadcast frames onto tactical timeline
    aligned_ball = np.full((n_tac, 3), np.nan, dtype=np.float32)
    aligned_dist = np.full((n_tac, 2), np.nan, dtype=np.float32)  # [dist_home_brd, dist_away_brd] in metres
    aligned_valid = np.full(n_tac, -1, dtype=np.int8)  # -1=outside broadcast, 0=TV cut/replay, 1=live wide cam

    # Pre-compute native broadcast player-to-ball distances (zero cross-camera homography error)
    brd_home_slots = np.flatnonzero(brd_slot == 0)
    brd_away_slots = np.flatnonzero(brd_slot == 1)
    brd_dist_h = np.full(n_brd, np.nan, dtype=np.float32)
    brd_dist_a = np.full(n_brd, np.nan, dtype=np.float32)
    vis_b = np.flatnonzero(brd_valid & np.isfinite(brd_ball[:, 0]) & np.isfinite(brd_ball[:, 1]))
    chunk = 10000
    for s in range(0, len(vis_b), chunk):
        idx = vis_b[s:s + chunk]
        p = np.asarray(brd_players[idx, :, :2], dtype=np.float32)
        b = np.asarray(brd_ball[idx, :2], dtype=np.float32)
        dx = (p[:, :, 0] - b[:, 0:1]) * PITCH_LENGTH
        dy = (p[:, :, 1] - b[:, 1:2]) * PITCH_WIDTH
        d = np.hypot(dx, dy)
        dh = np.where(np.isfinite(d[:, brd_home_slots]), d[:, brd_home_slots], np.inf)
        da = np.where(np.isfinite(d[:, brd_away_slots]), d[:, brd_away_slots], np.inf)
        min_h = np.min(dh, axis=1)
        min_a = np.min(da, axis=1)
        brd_dist_h[idx] = np.where(np.isfinite(min_h), min_h, np.nan)
        brd_dist_a[idx] = np.where(np.isfinite(min_a), min_a, np.nan)

    # Determine where the halftime split occurs in broadcast frames
    # Half 1 ends in tactical at h1_end -> broadcast frame ~ h1_end - delta1
    brd_h1_end = min(n_brd, max(0, h1_end - delta1 + 1000))
    brd_h2_start = max(0, min(n_brd, h2_start - delta2 - 1000))
    split_brd = (brd_h1_end + brd_h2_start) // 2

    # First mark all mapped tactical frames with base offset for valid=0/1
    for b_frame in range(n_brd):
        base_d = delta1 if b_frame < split_brd else delta2
        t_frame = b_frame + base_d
        if 0 <= t_frame < n_tac:
            aligned_valid[t_frame] = 1 if brd_valid[b_frame] else 0

    # Refine per valid broadcast segment and project broadcast_ball onto tactical frames
    valid_runs = runs_of(brd_valid)
    ball_diffs = []
    for b_start, b_stop in valid_runs:
        base_d = delta1 if b_start < split_brd else delta2
        seg_d = refine_segment_offset(tac_feat, brd_feat, b_start, b_stop, base_d, n_tac)
        for b_f in range(b_start, b_stop):
            t_f = b_f + seg_d
            if 0 <= t_f < n_tac:
                aligned_valid[t_f] = 1
                if np.isfinite(brd_ball[b_f, 0]):
                    aligned_ball[t_f] = brd_ball[b_f]
                    aligned_dist[t_f, 0] = brd_dist_h[b_f]
                    aligned_dist[t_f, 1] = brd_dist_a[b_f]
                    if np.isfinite(tac_ball[t_f, 0]):
                        dx = (tac_ball[t_f, 0] - brd_ball[b_f, 0]) * PITCH_LENGTH
                        dy = (tac_ball[t_f, 1] - brd_ball[b_f, 1]) * PITCH_WIDTH
                        ball_diffs.append(np.hypot(dx, dy))

    ball_diffs = np.array(ball_diffs)
    print("\nAlignment quality across {:,} co-visible ball frames:".format(len(ball_diffs)))
    print("  Median ball distance (Tactical vs Broadcast): {:.2f} m (p25={:.2f} m, p75={:.2f} m)".format(
        float(np.median(ball_diffs)), float(np.percentile(ball_diffs, 25)), float(np.percentile(ball_diffs, 75))
    ))

    # Check how many missing tactical ball frames are recovered on graded frames!
    graded = np.zeros(n_tac, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    tac_vis = np.isfinite(tac_ball[:, 0]) & graded
    brd_vis = np.isfinite(aligned_ball[:, 0]) & graded
    recovered = (~tac_vis) & brd_vis & graded
    combined_vis = (tac_vis | brd_vis) & graded

    print("\nBall visibility on the {:,} graded match frames:".format(int(graded.sum())))
    print("  Tactical only:            {:>7,} / {:,} ({:.1f}%)".format(
        int(tac_vis.sum()), int(graded.sum()), 100.0 * tac_vis.sum() / graded.sum()))
    print("  Recovered from Broadcast: {:>7,} new frames!".format(int(recovered.sum())))
    print("  Combined (Tac + Brd):     {:>7,} / {:,} ({:.1f}%)".format(
        int(combined_vis.sum()), int(graded.sum()), 100.0 * combined_vis.sum() / graded.sum()))

    tv_cuts_graded = (aligned_valid == 0) & graded
    print("  Broadcast TV cuts/replays (valid==0) on graded frames: {:>7,} frames ({:.1f}%)".format(
        int(tv_cuts_graded.sum()), 100.0 * tv_cuts_graded.sum() / graded.sum()
    ))

    np.save(os.path.join(CACHE_DIR, "aligned_broadcast_ball.npy"), aligned_ball)
    np.save(os.path.join(CACHE_DIR, "aligned_broadcast_dist.npy"), aligned_dist)
    np.save(os.path.join(CACHE_DIR, "aligned_broadcast_valid.npy"), aligned_valid)
    print("\nSaved aligned_broadcast_ball.npy, aligned_broadcast_dist.npy, aligned_broadcast_valid.npy")


if __name__ == "__main__":
    main()
