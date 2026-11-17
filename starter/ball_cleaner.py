"""Unsupervised multi-stage ball tracklet cleaner and stereo Tactical+Broadcast fuser.

Removes automatic tracking (ATD) ball false positives without using any ground-truth labels:
  1. Impossible teleport detours of any length (generalising the <= 5-frame blip filter).
  2. False out-of-bounds (OOB) locks on LED advertising boards during live play.
  3. Orphan slow/stationary locks on white pitch markings (penalty spots, centre spot, turf).
  4. Cross-camera Tactical vs Broadcast disagreement resolution (e.g. white-boot locks vs
     true pass trajectory), returning both cleaned tactical ball and cleaned fused ball.

All thresholds are physical football invariants (FIFA pitch markings, 35 m/s max flight speed)
and cross-camera/player kinematic consistency.
"""

import numpy as np

FIFA_MAX_BALL_MPS = 35.0
FIFA_PENALTY_MARK_M = 11.0
MIN_TRACKLET_BRIDGE_S = 4.0


def _runs_of_mask(mask):
    m = np.asarray(mask, dtype=bool)
    if len(m) == 0:
        return []
    pad = np.concatenate(([False], m, [False]))
    d = np.diff(pad.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _nearest_player_dist(xy_m, players_xy_m):
    """Per-frame Euclidean distance (m) from ball `xy_m` (N,2) to nearest finite player."""
    n = len(xy_m)
    out = np.full(n, np.nan, dtype=np.float32)
    if players_xy_m is None:
        return out
    vis = np.flatnonzero(np.isfinite(xy_m[:, 0]) & np.isfinite(xy_m[:, 1]))
    chunk = 10000
    for i0 in range(0, len(vis), chunk):
        idx = vis[i0:i0 + chunk]
        p = np.asarray(players_xy_m[idx, :, :2], dtype=np.float64)
        d = np.hypot(p[:, :, 0] - xy_m[idx, 0:1], p[:, :, 1] - xy_m[idx, 1:2])
        with np.errstate(all="ignore"):
            md = np.nanmin(d, axis=1)
        out[idx] = md.astype(np.float32)
    return out


def _split_tracklets(xy_m, max_speed_mps=FIFA_MAX_BALL_MPS, fps=25):
    """Split finite points of `xy_m` into tracklet frame-index arrays."""
    vis = np.flatnonzero(np.isfinite(xy_m[:, 0]) & np.isfinite(xy_m[:, 1]))
    if len(vis) == 0:
        return []
    tracklets = []
    start_i = 0
    for i in range(1, len(vis)):
        f0, f1 = int(vis[i - 1]), int(vis[i])
        dt = max(1, f1 - f0) / float(fps)
        dist = float(np.hypot(xy_m[f1, 0] - xy_m[f0, 0], xy_m[f1, 1] - xy_m[f0, 1]))
        if (f1 - f0) > 1 or (dist / dt) > max_speed_mps:
            tracklets.append(vis[start_i:i])
            start_i = i
    tracklets.append(vis[start_i:])
    return tracklets


def _remove_impossible_detours(xy_m, max_speed_mps=FIFA_MAX_BALL_MPS, fps=25,
                               max_detour_frames=75, max_bridge_s=MIN_TRACKLET_BRIDGE_S):
    """Iteratively drop short/medium tracklets flanked by >max_speed jumps when neighbors bridge."""
    out = xy_m.copy()
    removed = 0
    for _ in range(8):
        tracklets = _split_tracklets(out, max_speed_mps=max_speed_mps, fps=fps)
        if len(tracklets) < 3:
            break
        drop_idx = []
        for k in range(1, len(tracklets) - 1):
            cur = tracklets[k]
            if len(cur) > max_detour_frames:
                continue
            f_prev = int(tracklets[k - 1][-1])
            f_next = int(tracklets[k + 1][0])
            dt_in = (int(cur[0]) - f_prev) / float(fps)
            dt_out = (f_next - int(cur[-1])) / float(fps)
            dt_br = (f_next - f_prev) / float(fps)
            d_in = float(np.hypot(out[cur[0], 0] - out[f_prev, 0], out[cur[0], 1] - out[f_prev, 1]))
            d_out = float(np.hypot(out[f_next, 0] - out[cur[-1], 0], out[f_next, 1] - out[cur[-1], 1]))
            d_br = float(np.hypot(out[f_next, 0] - out[f_prev, 0], out[f_next, 1] - out[f_prev, 1]))
            v_in = d_in / max(dt_in, 1e-6)
            v_out = d_out / max(dt_out, 1e-6)
            v_br = d_br / max(dt_br, 1e-6)
            if v_in > max_speed_mps and v_out > max_speed_mps:
                if dt_br <= max_bridge_s and v_br <= max_speed_mps:
                    drop_idx.append(cur)
                elif len(cur) <= 5 and dt_in <= 1.0 and dt_out <= 1.0:
                    drop_idx.append(cur)
        if not drop_idx:
            break
        for cur in drop_idx:
            out[cur] = np.nan
            removed += len(cur)
    return out, removed


def _remove_false_oob_locks(xy_m, pdist, other_xy_m=None, other_pdist=None,
                            pitch=(105.3, 68.0), max_speed_mps=FIFA_MAX_BALL_MPS, fps=25):
    """Remove OOB detections caused by teleport jumps to ad-boards or contradicted by other cam."""
    out = xy_m.copy()
    L, W = pitch
    vis = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
    oob = vis & ((out[:, 0] < 0.0) | (out[:, 0] > L) | (out[:, 1] < 0.0) | (out[:, 1] > W))
    vis_idx = np.flatnonzero(vis)
    pos_in_vis = np.full(len(out), -1, dtype=np.int64)
    pos_in_vis[vis_idx] = np.arange(len(vis_idx))

    removed = 0
    for s, e in _runs_of_mask(oob):
        run_len = e - s
        # Check entry speed from previous visible frame (if in-bounds)
        vp = int(pos_in_vis[s])
        teleported_in = False
        if vp > 0:
            pf = int(vis_idx[vp - 1])
            if not oob[pf]:
                dt = (s - pf) / float(fps)
                d = float(np.hypot(out[s, 0] - out[pf, 0], out[s, 1] - out[pf, 1]))
                if dt <= 2.0 and (d / max(dt, 1e-6)) > max_speed_mps:
                    teleported_in = True

        # Check exit speed to next visible frame (if in-bounds)
        vp_end = int(pos_in_vis[e - 1])
        teleported_out = False
        if vp_end + 1 < len(vis_idx):
            nf = int(vis_idx[vp_end + 1])
            if not oob[nf]:
                dt = (nf - (e - 1)) / float(fps)
                d = float(np.hypot(out[nf, 0] - out[e - 1, 0], out[nf, 1] - out[e - 1, 1]))
                if dt <= 2.0 and (d / max(dt, 1e-6)) > max_speed_mps:
                    teleported_out = True

        mp = float(np.nanmin(pdist[s:e])) if np.isfinite(pdist[s:e]).any() else 99.0

        # Case 1: Teleported onto ad-board without a player nearby
        if (teleported_in or teleported_out) and mp > 3.0 and run_len <= 75:
            out[s:e] = np.nan
            removed += run_len
            continue

        # Case 2: Other camera clearly sees the ball well inside the pitch near a player
        if other_xy_m is not None and other_pdist is not None and mp > 4.0:
            ov = np.isfinite(other_xy_m[s:e, 0]) & np.isfinite(other_xy_m[s:e, 1])
            if ov.sum() >= max(2, run_len // 2):
                ox, oy = other_xy_m[s:e, 0][ov], other_xy_m[s:e, 1][ov]
                in_pitch = (ox > 2.0) & (ox < L - 2.0) & (oy > 2.0) & (oy < W - 2.0)
                op = other_pdist[s:e][ov]
                near_p = np.isfinite(op) & (op <= 3.5)
                if np.mean(in_pitch & near_p) >= 0.5:
                    out[s:e] = np.nan
                    removed += run_len
    return out, removed


def _remove_ghost_spot_locks(xy_m, pdist, sp_mask, other_xy_m=None,
                             pitch=(105.3, 68.0), fps=25):
    """Remove slow/stationary false locks isolated far from all players (marks, turf)."""
    out = xy_m.copy()
    L, W = pitch
    fifa_spots = np.array([
        [FIFA_PENALTY_MARK_M, 0.5 * W],
        [L - FIFA_PENALTY_MARK_M, 0.5 * W],
        [0.5 * L, 0.5 * W],
        [0.0, 0.0], [L, 0.0], [0.0, W], [L, W],
    ], dtype=np.float64)

    tracklets = _split_tracklets(out, max_speed_mps=FIFA_MAX_BALL_MPS, fps=fps)
    removed = 0
    for cur in tracklets:
        if len(cur) < 5:
            continue
        if sp_mask is not None and np.any(sp_mask[cur]):
            continue
        pts = out[cur]
        disp = float(np.hypot(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()))
        dur_s = max(1, int(cur[-1] - cur[0])) / float(fps)
        avg_drift_mps = disp / dur_s
        mp = float(np.nanmin(pdist[cur])) if np.isfinite(pdist[cur]).any() else 99.0

        # If other camera agrees within 3.0 m, don't drop it
        if other_xy_m is not None:
            ov = np.isfinite(other_xy_m[cur, 0]) & np.isfinite(other_xy_m[cur, 1])
            if ov.any():
                agree = np.hypot(pts[ov, 0] - other_xy_m[cur, 0][ov], pts[ov, 1] - other_xy_m[cur, 1][ov])
                if float(np.nanmedian(agree)) < 3.0:
                    continue

        # Stationary lock near a known FIFA white spot with no player nearby
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        near_fifa_spot = bool(np.min(np.hypot(fifa_spots[:, 0] - cx, fifa_spots[:, 1] - cy)) < 2.0)
        if near_fifa_spot and disp < 1.0 and mp > 3.0:
            out[cur] = np.nan
            removed += len(cur)
            continue

        # Orphan stationary/creeping lock far from every player over its entire lifespan
        if avg_drift_mps < 1.5 and mp > 4.5:
            out[cur] = np.nan
            removed += len(cur)

    return out, removed


def _resolve_stereo_disagreements(tac_m, brd_m, tac_pdist, brd_pdist,
                                  max_speed_mps=FIFA_MAX_BALL_MPS, fps=25):
    """Compare cleaned Tactical & Broadcast tracks; reject only discontinuous false locks.

    Smooth divergences between cameras (e.g. z > 0 aerial ball parallax) are preserved so
    downstream modules can use ||p_tac - p_brd|| as an aerial-flight indicator.
    """
    tac_out = tac_m.copy()
    brd_out = brd_m.copy()
    t_vis = np.isfinite(tac_out[:, 0]) & np.isfinite(tac_out[:, 1])
    b_vis = np.isfinite(brd_out[:, 0]) & np.isfinite(brd_out[:, 1])
    both = t_vis & b_vis
    diff = np.full(len(tac_out), np.nan, dtype=np.float32)
    diff[both] = np.hypot(tac_out[both, 0] - brd_out[both, 0], tac_out[both, 1] - brd_out[both, 1])

    disagree = both & (diff > 5.0)
    dropped_tac, dropped_brd = 0, 0

    for s, e in _runs_of_mask(disagree):
        if e - s < 3:
            continue

        # Check kinematic continuity at endpoints against nearby agreed frames (diff <= 4.0 m)
        if float(np.nanmedian(diff[s:e])) > 8.0:
            win = int(1.5 * fps)
            pre_agree = np.flatnonzero(both[max(0, s - win):s] & (diff[max(0, s - win):s] <= 4.0))
            post_agree = np.flatnonzero(both[e:min(len(tac_out), e + win)] & (diff[e:min(len(tac_out), e + win)] <= 4.0))
            err_t, err_b = 0.0, 0.0
            checks = 0
            if len(pre_agree):
                pf = max(0, s - win) + int(pre_agree[-1])
                dt = max(1, s - pf) / float(fps)
                vt = np.hypot(tac_out[s, 0] - tac_out[pf, 0], tac_out[s, 1] - tac_out[pf, 1]) / dt
                vb = np.hypot(brd_out[s, 0] - brd_out[pf, 0], brd_out[s, 1] - brd_out[pf, 1]) / dt
                err_t += vt
                err_b += vb
                checks += 1
            if len(post_agree):
                nf = e + int(post_agree[0])
                dt = max(1, nf - (e - 1)) / float(fps)
                vt = np.hypot(tac_out[nf, 0] - tac_out[e - 1, 0], tac_out[nf, 1] - tac_out[e - 1, 1]) / dt
                vb = np.hypot(brd_out[nf, 0] - brd_out[e - 1, 0], brd_out[nf, 1] - brd_out[e - 1, 1]) / dt
                err_t += vt
                err_b += vb
                checks += 1

            # Drop only if one camera clearly violates flight physics (> 35 m/s) while the other is smooth
            if checks > 0:
                if err_t / checks > max_speed_mps and err_b / checks <= 0.7 * max_speed_mps:
                    tac_out[s:e] = np.nan
                    dropped_tac += (e - s)
                elif err_b / checks > max_speed_mps and err_t / checks <= 0.7 * max_speed_mps:
                    brd_out[s:e] = np.nan
                    dropped_brd += (e - s)

    return tac_out, brd_out, dropped_tac, dropped_brd


def clean_and_fuse_ball(
    tac_ball, brd_ball=None, brd_valid=None, players=None, events=None,
    pitch=(105.3, 68.0), fps=25,
):
    """Clean Tactical & Broadcast ball tracks and return (cleaned_tac_norm, fused_norm, stats)."""
    L, W = float(pitch[0]), float(pitch[1])
    n = len(tac_ball)
    scale = np.array([L, W], dtype=np.float64)

    tac_m = np.asarray(tac_ball[:, :2], dtype=np.float64) * scale
    if brd_ball is not None:
        brd_m = np.asarray(brd_ball[:, :2], dtype=np.float64) * scale
        if brd_valid is not None:
            brd_m[np.asarray(brd_valid) != 1] = np.nan
    else:
        brd_m = np.full_like(tac_m, np.nan)

    players_m = np.asarray(players[:, :, :2], dtype=np.float64) * scale if players is not None else None

    sp_mask = np.zeros(n, dtype=bool)
    if events:
        for ev in events:
            if (ev.get("code") or "").strip().upper() == "SET PIECES":
                s = max(0, int(ev["start_frame"]) - fps)
                e = min(n, int(ev["end_frame"]) + fps)
                sp_mask[s:e] = True

    # Nearest player distances before cleaning
    tac_pdist = _nearest_player_dist(tac_m, players_m)
    brd_pdist = _nearest_player_dist(brd_m, players_m)

    # Stage 1: Remove teleport detours on each camera
    tac_m, t_detour = _remove_impossible_detours(tac_m, fps=fps)
    brd_m, b_detour = _remove_impossible_detours(brd_m, fps=fps)

    # Stage 2: Remove false OOB / ad-board locks
    tac_m, t_oob = _remove_false_oob_locks(tac_m, tac_pdist, brd_m, brd_pdist, pitch=pitch, fps=fps)
    brd_m, b_oob = _remove_false_oob_locks(brd_m, brd_pdist, tac_m, tac_pdist, pitch=pitch, fps=fps)

    # Stage 3: Remove isolated stationary/slow ghost locks (white spots, turf)
    tac_m, t_ghost = _remove_ghost_spot_locks(tac_m, tac_pdist, sp_mask, brd_m, pitch=pitch, fps=fps)
    brd_m, b_ghost = _remove_ghost_spot_locks(brd_m, brd_pdist, sp_mask, tac_m, pitch=pitch, fps=fps)

    # Stage 4: Stereo cross-camera disagreement resolution (prefer player-proximate / continuous track)
    tac_m, brd_m, t_stereo, b_stereo = _resolve_stereo_disagreements(
        tac_m, brd_m, tac_pdist, brd_pdist, fps=fps
    )

    # Stage 5: Fuse (prefer cleaned Tactical; fill gaps with Broadcast) + detour sweep
    fused_m = tac_m.copy()
    tac_vis = np.isfinite(tac_m[:, 0]) & np.isfinite(tac_m[:, 1])
    brd_vis = np.isfinite(brd_m[:, 0]) & np.isfinite(brd_m[:, 1])
    fill = (~tac_vis) & brd_vis
    fused_m[fill] = brd_m[fill]

    fused_m, f_detour = _remove_impossible_detours(fused_m, fps=fps)

    stats = {
        "tac_removed_detour": int(t_detour),
        "tac_removed_oob": int(t_oob),
        "tac_removed_ghost": int(t_ghost),
        "tac_removed_stereo": int(t_stereo),
        "brd_removed_detour": int(b_detour),
        "brd_removed_oob": int(b_oob),
        "brd_removed_ghost": int(b_ghost),
        "brd_removed_stereo": int(b_stereo),
        "fused_removed_detour": int(f_detour),
        "fused_filled_from_brd": int(fill.sum()),
    }
    return (tac_m / scale).astype(np.float64), (fused_m / scale).astype(np.float64), stats
