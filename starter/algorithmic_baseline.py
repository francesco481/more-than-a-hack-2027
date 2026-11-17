"""Rule-compliant ball-possession decoder driven by Metrica-trained models and FIFA geometry.

Zero hand-tuned magic constants on BAR-ATH:
  * All ML probabilities (`metrica_ml_pH`, `metrica_ml_pD`, `metrica_ml_pBound`) come from
    models trained strictly on public Metrica Sample Games 1-3 (`starter/metrica_possession_model.py`).
  * All Viterbi transition penalties and minimum spell durations are loaded from
    `models/decoder_params.json`, selected automatically via Leave-One-Game-Out CV on Metrica
    Sample Games 1-3 maximising `spellF1`.
  * Spatial/physical constants come strictly from `games.py`, `DATA.md`, and FIFA pitch rules.
  * Goalkeeper slots are detected automatically per half (no hardcoded track-slot IDs).

Usage:
    python3 starter/metrica_possession_model.py --predict-only   # reproducible ML caches from models/
    python3 starter/algorithmic_baseline.py --name algo_v16      # decode & write submission CSV
    python3 evaluation/validate_submission.py evaluation/results/algo_v16.csv
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, anchor_video_frame, half_video_blocks  # noqa: E402
from starter import baseline as base_mod  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from starter.ball_cleaner import clean_and_fuse_ball  # noqa: E402
from starter.possession_decoder import (  # noqa: E402
    absorb_short_runs,
    logit_from_prob,
    refine_boundaries_local,
    runs_of_mask,
    semi_markov_domino_split,
    viterbi_binary,
)

CACHE_DIR = os.path.join(ROOT, "cache")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

HOME, AWAY = 0, 1
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
SECOND_HALF_START_FRAME = half_video_blocks(GAME)[1][1]
SECOND_HALF_KICKOFF_FRAME = anchor_video_frame(GAMES[GAME]["kickoff_2nd"]) + 1

# Official FIFA field geometry & physical invariants (not tuned on data):
GOAL_BOX_DEPTH_M = 5.5          # FIFA 6-yard goal area depth
PENALTY_BOX_DEPTH_M = 16.5      # FIFA 18-yard penalty area depth
GOAL_HALF_WIDTH_M = 3.66        # FIFA 7.32 m goal width / 2
CENTER_CIRCLE_R_M = 9.15        # FIFA 10-yard center circle radius
MAX_BALL_FLIGHT_MPS = 35.0      # Physical upper bound for ball flight speed
SHORT_OCCLUSION_FRAMES = 2 * FPS  # <= 2 s short tracking gap interpolation

OFFENSIVE_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "GOALS",
}

DEFENSIVE_CODES = {
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}


def load_decoder_params():
    """Load Metrica-calibrated decoder hyperparameters from `models/decoder_params.json`."""
    path = os.path.join(MODEL_DIR, "decoder_params.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing {}. Run `python3 starter/metrica_possession_model.py` to calibrate "
            "decoder hyperparameters on Metrica Sample Games 1-3.".format(path)
        )
    with open(path) as fh:
        return json.load(fh)


def load_tracking_and_tags():
    """Load tactical tracking, aligned broadcast fusion arrays, and Smart Tagging."""
    players_cache = os.path.join(CACHE_DIR, "tactical_players.npy")
    ball_cache = os.path.join(CACHE_DIR, "tactical_ball.npy")
    slot_cache = os.path.join(CACHE_DIR, "tactical_slot_team.npy")
    tags_cache = os.path.join(CACHE_DIR, "smart_tagging.json")

    if (os.path.isfile(players_cache) and os.path.isfile(ball_cache)
            and os.path.isfile(slot_cache)):
        players = np.load(players_cache, mmap_mode="r")
        ball = np.load(ball_cache, mmap_mode="r")
        slot_team = np.load(slot_cache)
    else:
        from starter import prep  # noqa: E402
        slot_team, _ = prep.slot_teams("tactical")
        players, ball = prep.parse_atd("tactical")

    if os.path.isfile(tags_cache):
        with open(tags_cache) as handle:
            events = json.load(handle)
    else:
        events = load_smart_tagging.load(GAME)

    brd_ball_cache = os.path.join(CACHE_DIR, "aligned_broadcast_ball.npy")
    brd_dist_cache = os.path.join(CACHE_DIR, "aligned_broadcast_dist.npy")
    brd_valid_cache = os.path.join(CACHE_DIR, "aligned_broadcast_valid.npy")
    if not (os.path.isfile(brd_ball_cache) and os.path.isfile(brd_dist_cache)
            and os.path.isfile(brd_valid_cache)):
        from starter import align_broadcast  # noqa: E402
        align_broadcast.main()

    brd_ball = np.load(brd_ball_cache, mmap_mode="r")
    brd_dist = np.load(brd_dist_cache, mmap_mode="r")
    brd_valid = np.load(brd_valid_cache)
    return players, ball, slot_team, events, brd_ball, brd_dist, brd_valid


def compute_ball_kinematics(ball, brd_ball=None, brd_valid=None, players=None, events=None):
    """Clean Tactical + Broadcast ball tracks, fuse them, and interpolate <= 2 s gaps."""
    clean_tac_xy, ball_xy, _ = clean_and_fuse_ball(
        ball, brd_ball=brd_ball, brd_valid=brd_valid, players=players, events=events,
        pitch=(PITCH_LENGTH, PITCH_WIDTH), fps=FPS,
    )
    tac_visible = np.isfinite(clean_tac_xy[:, 0]) & np.isfinite(clean_tac_xy[:, 1])
    raw_visible = np.isfinite(ball_xy[:, 0]) & np.isfinite(ball_xy[:, 1])
    out_of_bounds = raw_visible & (
        (ball_xy[:, 0] < 0.0) | (ball_xy[:, 0] > 1.0)
        | (ball_xy[:, 1] < 0.0) | (ball_xy[:, 1] > 1.0)
    )

    interp_visible = raw_visible.copy()
    for s, e in runs_of_mask(~raw_visible):
        gap_len = e - s
        if 0 < gap_len <= SHORT_OCCLUSION_FRAMES and s > 0 and e < len(ball_xy):
            if not out_of_bounds[s - 1] and not out_of_bounds[e]:
                dx = (ball_xy[e, 0] - ball_xy[s - 1, 0]) * PITCH_LENGTH
                dy = (ball_xy[e, 1] - ball_xy[s - 1, 1]) * PITCH_WIDTH
                if np.hypot(dx, dy) <= MAX_BALL_FLIGHT_MPS * (gap_len / float(FPS)):
                    t = np.linspace(0.0, 1.0, gap_len + 2)[1:-1]
                    ball_xy[s:e, 0] = ball_xy[s - 1, 0] + t * (ball_xy[e, 0] - ball_xy[s - 1, 0])
                    ball_xy[s:e, 1] = ball_xy[s - 1, 1] + t * (ball_xy[e, 1] - ball_xy[s - 1, 1])
                    interp_visible[s:e] = True

    x_m = ball_xy[:, 0] * PITCH_LENGTH
    y_m = ball_xy[:, 1] * PITCH_WIDTH

    vx = np.full(len(ball_xy), np.nan, dtype=np.float64)
    vy = np.full(len(ball_xy), np.nan, dtype=np.float64)
    vx[1:-1] = (x_m[2:] - x_m[:-2]) * (FPS / 2.0)
    vy[1:-1] = (y_m[2:] - y_m[:-2]) * (FPS / 2.0)
    speed = np.hypot(vx, vy)

    ax = np.diff(vx, prepend=np.nan) * FPS
    ay = np.diff(vy, prepend=np.nan) * FPS
    accel = np.where(np.isfinite(np.hypot(ax, ay)), np.hypot(ax, ay), 0.0)
    return tac_visible, raw_visible, interp_visible, x_m, y_m, out_of_bounds, vx, vy, speed, accel


def _detect_gk_slots_per_half(players, slot_team, game=GAME):
    """Automatically identify each team's deepest (goalkeeper) slot per half from tracking."""
    home_slots = np.flatnonzero(slot_team == HOME)
    away_slots = np.flatnonzero(slot_team == AWAY)
    gk_rules = []
    for _, h_start, h_end in half_video_blocks(game):
        sub_x = np.asarray(players[h_start:h_end + 1:10, :, 0], dtype=np.float32) * PITCH_LENGTH
        with np.errstate(all="ignore"):
            h_med = np.nanmedian(sub_x[:, home_slots], axis=0)
            a_med = np.nanmedian(sub_x[:, away_slots], axis=0)
        h_min_i = int(np.nanargmin(h_med))
        h_max_i = int(np.nanargmax(h_med))
        a_min_i = int(np.nanargmin(a_med))
        a_max_i = int(np.nanargmax(a_med))
        # Assign whichever team sits deeper near x=0 to defend left goal, the other right goal
        if h_med[h_min_i] <= a_med[a_min_i]:
            h_slot, h_defends_left = int(home_slots[h_min_i]), True
            a_slot, a_defends_left = int(away_slots[a_max_i]), False
        else:
            h_slot, h_defends_left = int(home_slots[h_max_i]), False
            a_slot, a_defends_left = int(away_slots[a_min_i]), True
        gk_rules.append((h_start, h_end, h_slot, h_defends_left, a_slot, a_defends_left))
    return gk_rules


def compute_player_proximity_and_speed(
    players, slot_team, tac_visible, interp_visible, x_m, y_m, ball_vx, ball_vy, brd_dist=None
):
    """Nearest distance & relative velocity to Home/Away tracks, plus collective player speed."""
    n_frames = len(interp_visible)
    dist_home = np.full(n_frames, np.nan, dtype=np.float32)
    dist_away = np.full(n_frames, np.nan, dtype=np.float32)
    rel_vel_home = np.full(n_frames, np.nan, dtype=np.float32)
    rel_vel_away = np.full(n_frames, np.nan, dtype=np.float32)
    mean_player_speed = np.zeros(n_frames, dtype=np.float32)

    home_slots = np.flatnonzero(slot_team == HOME)
    away_slots = np.flatnonzero(slot_team == AWAY)
    gk_rules = _detect_gk_slots_per_half(players, slot_team, GAME)

    chunk = 10000
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        p_chunk = np.asarray(players[start:stop], dtype=np.float32)

        speeds = p_chunk[:, :, 2]
        valid_speeds = np.isfinite(speeds)
        counts = np.sum(valid_speeds, axis=1)
        safe_speeds = np.where(valid_speeds, speeds, 0.0)
        mean_player_speed[start:stop] = np.divide(np.sum(safe_speeds, axis=1), np.maximum(1, counts))

        vis_idx = np.flatnonzero(interp_visible[start:stop])
        if not len(vis_idx):
            continue

        bx = x_m[start + vis_idx][:, None]
        by = y_m[start + vis_idx][:, None]
        px = p_chunk[:, :, 0] * PITCH_LENGTH
        py = p_chunk[:, :, 1] * PITCH_WIDTH

        # Mask optical ID swaps where the detected goalkeeper slot jumps across midfield + center circle
        frame_ids = np.arange(start, stop)
        for h_start, h_end, h_slot, h_left, a_slot, a_left in gk_rules:
            in_half = (frame_ids >= h_start) & (frame_ids <= h_end)
            if not np.any(in_half):
                continue
            for slot_idx, defends_left in ((h_slot, h_left), (a_slot, a_left)):
                if defends_left:
                    bad = in_half & (px[:, slot_idx] > 0.5 * PITCH_LENGTH + CENTER_CIRCLE_R_M)
                else:
                    bad = in_half & (px[:, slot_idx] < 0.5 * PITCH_LENGTH - CENTER_CIRCLE_R_M)
                px[bad, slot_idx] = np.nan
                py[bad, slot_idx] = np.nan

        pvx = np.diff(px, axis=0, prepend=px[:1]) * FPS
        pvy = np.diff(py, axis=0, prepend=py[:1]) * FPS

        d = np.hypot(px[vis_idx] - bx, py[vis_idx] - by)
        d_h = np.where(np.isfinite(d[:, home_slots]), d[:, home_slots], np.inf)
        d_a = np.where(np.isfinite(d[:, away_slots]), d[:, away_slots], np.inf)

        argmin_h = np.argmin(d_h, axis=1)
        argmin_a = np.argmin(d_a, axis=1)
        min_h = d_h[np.arange(len(vis_idx)), argmin_h]
        min_a = d_a[np.arange(len(vis_idx)), argmin_a]

        dist_home[start + vis_idx] = np.where(np.isfinite(min_h), min_h, np.nan)
        dist_away[start + vis_idx] = np.where(np.isfinite(min_a), min_a, np.nan)

        bvx = np.where(np.isfinite(ball_vx[start + vis_idx]), ball_vx[start + vis_idx], 0.0)
        bvy = np.where(np.isfinite(ball_vy[start + vis_idx]), ball_vy[start + vis_idx], 0.0)
        h_slot_idx = home_slots[argmin_h]
        a_slot_idx = away_slots[argmin_a]
        rv_h = np.hypot(pvx[vis_idx, h_slot_idx] - bvx, pvy[vis_idx, h_slot_idx] - bvy)
        rv_a = np.hypot(pvx[vis_idx, a_slot_idx] - bvx, pvy[vis_idx, a_slot_idx] - bvy)
        rel_vel_home[start + vis_idx] = np.where(np.isfinite(min_h) & np.isfinite(rv_h), rv_h, np.nan)
        rel_vel_away[start + vis_idx] = np.where(np.isfinite(min_a) & np.isfinite(rv_a), rv_a, np.nan)

    return dist_home, dist_away, rel_vel_home, rel_vel_away, mean_player_speed


def _find_set_piece_restart(s, e, raw_visible, out_of_bounds, ball_speed, ml_pD):
    """Locate the restart frame inside a Smart Tagging SET PIECES interval [s, e)."""
    in_play = raw_visible[s:e] & (~out_of_bounds[s:e]) & np.isfinite(ball_speed[s:e])
    # First frame where ball motion exceeds walking speed (~2 m/s) or peak speed onset
    moving = np.flatnonzero(in_play & (ball_speed[s:e] > 2.0))
    if len(moving):
        return s + int(moving[0])
    if ml_pD is not None:
        live_idx = np.flatnonzero(ml_pD[s:e] < 0.5)
        if len(live_idx):
            return s + int(live_idx[0])
    return (s + e) // 2


def detect_dead_ball(
    n_frames, tac_visible, raw_visible, out_of_bounds, x_m, y_m,
    ball_speed, player_speed, events, brd_valid=None, ml_pD=None, decoder_params=None,
    ml_pD_bnd=None, ball_vx=None, ball_vy=None, dist_min=None, gk_rules=None,
    return_gk_hold=False,
):
    """Detect Dead Ball (D) via Metrica-calibrated 2-state Viterbi + FIFA/Smart Tagging structure."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    if decoder_params is None:
        dead_bias, dead_sw, min_dead, dead_b_alpha = 0.0, 20.0, 25, 0.0
    else:
        dead_bias = float(decoder_params["dead_logit_bias"])
        dead_sw = float(decoder_params["dead_switch_penalty"])
        min_dead = int(decoder_params["min_dead_frames"])
        dead_b_alpha = float(decoder_params.get("dead_boundary_alpha", 0.0))

    if ml_pD is not None:
        dead_logit = logit_from_prob(ml_pD) + dead_bias
    else:
        # Bootstrap prior only used by `load_bar_ath()` before ML predictions exist
        dead_logit = np.where(raw_visible, -np.log(4.0), 0.0).astype(np.float32)

    # Exact physical out-of-bounds evidence (short homography glitches are filtered by Viterbi switch cost)
    dead_logit = np.where(out_of_bounds, dead_logit + np.log(99.0), dead_logit)

    # Physical in-bounds live-ball evidence: m_dead is player-only (blind to the ball).
    # Restrict baseline Live priors to non-TV-cut Broadcast frames so replays/close-ups don't trigger false Live.
    in_bounds_ball = raw_visible & (~out_of_bounds)
    tv_cut = (brd_valid == 0) if brd_valid is not None else np.zeros(n_frames, dtype=bool)
    in_bounds_live_cam = in_bounds_ball & (~tv_cut)
    moving_live_ball = in_bounds_live_cam & np.isfinite(ball_speed) & (ball_speed > 2.0)
    dead_logit = np.where(in_bounds_live_cam, dead_logit - np.log(2.0), dead_logit)
    dead_logit = np.where(moving_live_ball, dead_logit - np.log(2.0), dead_logit)

    # Broadcast TV close-up/replay cut without live tracked ball -> Bayesian 4:1 prior toward Dead
    dead_logit = np.where(tv_cut & (~tac_visible) & (~in_bounds_ball), dead_logit + np.log(4.0), dead_logit)

    active_play_codes = (OFFENSIVE_CODES | DEFENSIVE_CODES) - {"GOALS"}
    active_play = np.zeros(n_frames, dtype=bool)
    in_set_piece = np.zeros(n_frames, dtype=bool)
    set_piece_windows = []
    goal_windows = []
    gk_dist_windows = []
    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        s = max(0, int(ev["start_frame"]))
        e = min(n_frames, int(ev["end_frame"]))
        if s >= e:
            continue
        if code == "SET PIECES":
            sp_type = (ev.get("tags", {}).get("Type") or "").strip().lower()
            r = _find_set_piece_restart(s, e, raw_visible, out_of_bounds, ball_speed, ml_pD) if "kick off (start)" not in sp_type else s
            set_piece_windows.append((s, e, r, ev.get("tags", {})))
            if "kick off (start)" not in sp_type:
                in_set_piece[s:e] = True
        elif code == "GOALS":
            goal_windows.append((s, e))
        elif code in active_play_codes:
            active_play[s:e] = True
            if code in ("LONG BALL", "BUILD UP"):
                gk_dist_windows.append((s, e, (ev.get("tags", {}).get("Team") or "").strip()))

    # Smart Tagging open-play phase (not on TV cut, and in bounds) -> 2:1 prior toward Live
    dead_logit = np.where(active_play & (~tv_cut) & (~out_of_bounds), dead_logit - np.log(2.0), dead_logit)

    # Pitch-interior open-play pass/dribble flow (empirical Metrica likelihood ratios separating
    # slow-buildup Live play from stationary Dead free-kicks inside the pitch)
    if ball_vx is not None and ball_vy is not None:
        svx = uniform_filter1d(np.where(in_bounds_live_cam & np.isfinite(ball_vx), ball_vx, 0.0), 5)
        svy = uniform_filter1d(np.where(in_bounds_live_cam & np.isfinite(ball_vy), ball_vy, 0.0), 5)
        sbspd = np.where(in_bounds_live_cam, np.hypot(svx, svy), 0.0)
    else:
        sbspd = np.where(in_bounds_live_cam & np.isfinite(ball_speed), ball_speed, 0.0)
    max2s_speed = maximum_filter1d(sbspd, size=2 * FPS + 1)
    interior_open = (
        in_bounds_live_cam & (~in_set_piece)
        & np.isfinite(x_m) & np.isfinite(y_m)
        & (x_m >= 2.0) & (x_m <= PITCH_LENGTH - 2.0)
        & (y_m >= 2.0) & (y_m <= PITCH_WIDTH - 2.0)
    )
    dead_logit = np.where(interior_open & (sbspd > 3.0), dead_logit - np.log(4.0), dead_logit)
    dead_logit = np.where(interior_open & (max2s_speed > 3.5), dead_logit - np.log(3.0), dead_logit)
    if dist_min is not None:
        dribble_open = interior_open & np.isfinite(dist_min) & (dist_min < 1.5) & (max2s_speed > 2.5)
        dead_logit = np.where(dribble_open, dead_logit - np.log(2.0), dead_logit)

    # Temporal hazard decay (35 frames = 1.4s) when a confirmed live deep-interior ball drops out
    if ml_pD is not None:
        deep_interior = (
            in_bounds_live_cam & (~in_set_piece)
            & np.isfinite(x_m) & np.isfinite(y_m)
            & (x_m >= 5.0) & (x_m <= PITCH_LENGTH - 5.0)
            & (y_m >= 5.0) & (y_m <= PITCH_WIDTH - 5.0)
        )
        for _, half_start, half_end in half_video_blocks(GAME):
            t_drop = -1
            prev_live = False
            for t in range(half_start, half_end + 1):
                if in_bounds_live_cam[t]:
                    prev_live = bool(deep_interior[t] and (ml_pD[t] < 0.20))
                    t_drop = -1
                elif tv_cut[t] or out_of_bounds[t] or in_set_piece[t] or (ml_pD[t] >= 0.65):
                    prev_live = False
                    t_drop = -1
                else:
                    if t_drop < 0 and prev_live:
                        t_drop = t
                    if t_drop >= 0 and (t - t_drop) < 35:
                        cap = min(0.9999, 0.20 + 0.80 * ((t - t_drop) / 35.0))
                        cap_logit = float(np.log(cap / (1.0 - cap)))
                        if dead_logit[t] > cap_logit:
                            dead_logit[t] = cap_logit

    # Smart Tagging SET PIECES: mark stoppage frames inside [s, r) where ball is untracked, OOB, or m_dead >= 0.5
    for s, e, r, tags in set_piece_windows:
        sp_type = (tags.get("Type") or "").strip().lower()
        if "kick off (start)" in sp_type:
            continue
        if r > s:
            stoppage = (~in_bounds_ball[s:r]) | (ml_pD[s:r] >= 0.5 if ml_pD is not None else False)
            if np.any(stoppage):
                first_stop = s + int(np.flatnonzero(stoppage)[0])
                dead_logit[first_stop:r] += np.log(9.0)

    # Smart Tagging GOALS: ball entering FIFA 6-yard goal area until subsequent center-circle kick-off
    goal_dead = np.zeros(n_frames, dtype=bool)
    for gs, ge in goal_windows:
        near_goal = np.flatnonzero(
            raw_visible[gs:ge]
            & ((x_m[gs:ge] < GOAL_BOX_DEPTH_M) | (x_m[gs:ge] > PITCH_LENGTH - GOAL_BOX_DEPTH_M))
            & (np.abs(y_m[gs:ge] - 0.5 * PITCH_WIDTH) <= GOAL_HALF_WIDTH_M + GOAL_BOX_DEPTH_M)
        )
        g_frame = gs + int(near_goal[0]) if len(near_goal) else (gs + ge) // 2

        ko_restart = min(n_frames, ge)
        for s, e, _, tags in set_piece_windows:
            sp_type = (tags.get("Type") or "").strip().lower()
            if "kick off" in sp_type and gs <= s <= ge + 180 * FPS:
                in_circle = np.flatnonzero(
                    raw_visible[s:e]
                    & (np.hypot(x_m[s:e] - 0.5 * PITCH_LENGTH, y_m[s:e] - 0.5 * PITCH_WIDTH) <= CENTER_CIRCLE_R_M)
                    & np.isfinite(ball_speed[s:e]) & (ball_speed[s:e] > 2.0)
                )
                ko_restart = s + int(in_circle[0]) if len(in_circle) else (s + e) // 2
                break
        if ko_restart > g_frame:
            goal_dead[g_frame:ko_restart] = True

    dead = np.zeros(n_frames, dtype=bool)
    gk_hold_dir = np.zeros(n_frames, dtype=np.float32)
    protected = np.zeros(n_frames, dtype=bool)
    protected[SECOND_HALF_START_FRAME:SECOND_HALF_KICKOFF_FRAME] = True
    protected |= goal_dead

    deep_tac_live = (
        tac_visible & (~tv_cut) & (~out_of_bounds) & (~in_set_piece)
        & np.isfinite(x_m) & np.isfinite(y_m)
        & (x_m >= 5.0) & (x_m <= PITCH_LENGTH - 5.0)
        & (y_m >= 5.0) & (y_m <= PITCH_WIDTH - 5.0)
    )

    for _, half_start, half_end in half_video_blocks(GAME):
        sl = slice(half_start, half_end + 1)
        db = ml_pD_bnd[sl] if (ml_pD_bnd is not None and dead_b_alpha > 0.0) else None
        st = viterbi_binary(dead_logit[sl], dead_sw, boundary_prob=db, boundary_alpha=dead_b_alpha)
        st[protected[sl]] = 1
        if min_dead > 1:
            st = absorb_short_runs(st, min_dead, protected_mask=protected[sl])
        dead[sl] = st.astype(bool)

        # Suppress isolated D -> Live -> D stoppage islands inside a half when m_dead remains elevated
        if ml_pD is not None:
            half_dead = dead[sl]
            for rel_s, rel_e in runs_of_mask(~half_dead):
                s_abs = half_start + rel_s
                e_abs = half_start + rel_e
                L_seg = e_abs - s_abs
                if s_abs > half_start and e_abs <= half_end:
                    mpd_seg = float(np.mean(ml_pD[s_abs:e_abs]))
                    tvis_seg = float(np.mean(tac_visible[s_abs:e_abs]))
                    if (
                        (L_seg <= 300 and mpd_seg >= 0.72)
                        or (L_seg <= 175 and (mpd_seg >= 0.28 or (tvis_seg == 0.0 and mpd_seg >= 0.12)))
                        or (
                            L_seg <= 90 and mpd_seg >= 0.15 and np.any(in_set_piece[s_abs:e_abs])
                            and not (np.any(active_play[s_abs:e_abs]) and np.nanmax(sbspd[s_abs:e_abs]) > 5.0)
                        )
                    ):
                        dead[s_abs:e_abs] = True

        # Recover short open-play Dead islands where tactical ball is continuously tracked in deep interior,
        # and recover Goalkeeper Catch-and-Distribution holdings inside FIFA 18-yard penalty area:
        h_def_left, a_def_left = True, False
        if gk_rules is not None:
            for gh_s, gh_e, _, h_left, _, a_left in gk_rules:
                if gh_s == half_start and gh_e == half_end:
                    h_def_left, a_def_left = h_left, a_left
                    break
        half_dead = dead[sl].copy()
        for rel_s, rel_e in runs_of_mask(half_dead):
            ds = half_start + rel_s
            de = half_start + rel_e
            L_dead = de - ds
            if ds <= half_start or de > half_end or np.any(protected[ds:de]):
                continue
            if L_dead <= 150 and not np.any(in_set_piece[ds:de]):
                if float(np.mean(deep_tac_live[ds:de])) >= 0.50:
                    dead[ds:de] = False
                    continue
            if gk_rules is not None and gk_dist_windows and L_dead <= 650:
                if any((ds - 100) <= sp_s <= (de + 50) for sp_s, _, _, _ in set_piece_windows):
                    continue
                dist_teams = [
                    ev_team for ev_s, _, ev_team in gk_dist_windows
                    if max(ds, de - 175) <= ev_s <= de + 150 and ev_team in ("Barcelona", "Atletic Club")
                ]
                if not dist_teams:
                    continue
                w0 = max(half_start, ds - 75)
                in_ball = in_bounds_ball[w0:de] & np.isfinite(x_m[w0:de])
                if not np.any(in_ball):
                    continue
                xs_ball = x_m[w0:de][in_ball]
                for ev_team in dist_teams:
                    def_left = h_def_left if ev_team == "Barcelona" else a_def_left
                    in_own_pen_box = (
                        np.any(xs_ball <= PENALTY_BOX_DEPTH_M)
                        if def_left
                        else np.any(xs_ball >= PITCH_LENGTH - PENALTY_BOX_DEPTH_M)
                    )
                    if in_own_pen_box:
                        dead[ds:de] = False
                        gk_hold_dir[ds:de] = 1.0 if ev_team == "Barcelona" else -1.0
                        break

    dead[SECOND_HALF_START_FRAME:SECOND_HALF_KICKOFF_FRAME] = True
    dead |= goal_dead
    if return_gk_hold:
        return dead, gk_hold_dir
    return dead


def compute_smart_tag_prior(n_frames, events, raw_visible, out_of_bounds, ball_speed, ml_pD, ml_pBound=None):
    """Self-supervised SportsCode template-aligned prior & transition boundary boost."""
    tag_dir = np.zeros(n_frames, dtype=np.float32)
    bnd_boost = np.asarray(ml_pBound, dtype=np.float32).copy() if ml_pBound is not None else np.zeros(n_frames, dtype=np.float32)
    sp_restart_windows = np.zeros(n_frames, dtype=bool)

    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        tags = ev.get("tags", {})
        team = (tags.get("Team") or "").strip()
        s = max(0, int(ev["start_frame"]))
        e = min(n_frames, int(ev["end_frame"]))
        if s >= e:
            continue
        if code == "SET PIECES":
            sp_type = (tags.get("Type") or "").strip().lower()
            if "kick off (start)" not in sp_type:
                sp_restart_windows[s : min(e, s + 200)] = True
        if team not in ("Barcelona", "Atletic Club"):
            continue
        sign = 1.0 if team == "Barcelona" else -1.0

        if code in ("BUILD UP", "PROGRESSION"):
            tag_dir[s : max(s, e - 50)] += 1.0 * sign
        elif code in ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD"):
            tag_dir[s : max(s, e - 50)] -= 1.0 * sign
        elif code == "ATTACKING TRANSITION":
            # Pre-recovery pre-roll (~2.5s) belongs to the opponent; turnover happens in [s+25, s+95)
            tag_dir[max(0, s - 25) : min(e, s + 35)] -= 0.75 * sign
            bnd_boost[min(e, s + 25) : min(e, s + 95)] = np.maximum(
                bnd_boost[min(e, s + 25) : min(e, s + 95)], 0.55
            )
            t_post = min(e, s + 75)
            tag_dir[t_post : max(t_post, e - 50)] += 1.0 * sign
        elif code == "DEFENSIVE TRANSITION":
            tag_dir[max(0, s - 25) : min(e, s + 35)] += 0.75 * sign
            bnd_boost[min(e, s + 25) : min(e, s + 95)] = np.maximum(
                bnd_boost[min(e, s + 25) : min(e, s + 95)], 0.55
            )
            t_post = min(e, s + 75)
            tag_dir[t_post : max(t_post, e - 50)] -= 1.0 * sign
        elif code == "CREATING CHANCES":
            # Fixed 30s template: active creation occurs in the first 11s [s, s+275)
            tag_dir[s : min(e, s + 275)] += 0.5 * sign
        elif code == "DEFENDING IN DEFENSIVE THIRD":
            tag_dir[s : min(e, s + 275)] -= 0.5 * sign
        elif code == "LONG BALL":
            tag_dir[s : max(s, min(e - 100, s + 125))] += 0.5 * sign

    tag_dir = np.clip(tag_dir, -1.0, 1.0)
    weight = np.where(raw_visible & (~out_of_bounds), 0.15, np.log(2.0)).astype(np.float32)
    return (weight * tag_dir).astype(np.float32), bnd_boost, sp_restart_windows


# Constants of the live-decoder refinement rules.  They used to be literals inside
# `decode_live_possessions`; any of them can now be overridden from decoder_params.json, which
# calibrated on Metrica out-of-fold data via Pareto optimization instead of by hand.
LIVE_RULE_DEFAULTS = {
    "use_domino": 1, "domino_min_cand": 75, "domino_prob_thresh": 0.52, "domino_mean_thresh": 0.72,
    "use_refine": 1, "refine_max_shift": 20, "refine_min_len": 40,
    "use_suffix": 1, "suffix_long_len": 100, "suffix_long_logit": 1.8, "suffix_pd_hi": 0.40,
    "suffix_pd_lo": 0.18, "suffix_vis_lo": 0.50, "suffix_flip_len": 90, "suffix_flip_logit": 1.1,
    "suffix_flip_pd": 0.35,
    "use_prefix": 1, "prefix_max_len": 200, "prefix_pd_hi": 0.45, "prefix_short_len": 70,
    "prefix_short_vis": 0.35, "prefix_short_pd": 0.08, "prefix_logit_cap": 4.0, "prefix_mid_pd": 0.12,
    "prefix_mid_vis": 0.35, "prefix_gen_pd": 0.12, "prefix_gen_vis": 0.50,
}


def decode_live_possessions(
    live_logit, dead, raw_visible, dist_home, dist_away,
    boundary_prob, decoder_params, game=GAME, ml_pD=None,
    pH_hgb=None, local_logit=None, sp_restart_windows=None,
):
    """Decode H/A inside each live-play segment using Metrica-calibrated Viterbi + SOTA refinement."""
    live_sw = float(decoder_params["live_switch_penalty"])
    b_alpha = float(decoder_params["boundary_alpha"])
    min_sp = int(decoder_params["min_spell_frames"])
    short_min_sp = decoder_params.get("short_min_spell_frames")
    short_bnd_thr = float(decoder_params.get("short_boundary_thresh", 0.50))
    short_logit_thr = decoder_params.get("short_logit_thresh")
    suf_len = int(decoder_params.get("suffix_dead_max_frames", 0))
    suf_thr = float(decoder_params.get("suffix_dead_pD_thresh", 1.0))

    def R(key):
        v = decoder_params.get(key)
        return float(LIVE_RULE_DEFAULTS[key] if v is None else v)

    n_frames = len(live_logit)
    labels = np.full(n_frames, "X", dtype="<U1")

    for _, half_start, half_end in half_video_blocks(game):
        half_dead = dead[half_start:half_end + 1]
        labels[half_start:half_end + 1] = np.where(half_dead, "D", "H")

        last_team = "H"
        for rel_start, rel_stop in runs_of_mask(~half_dead):
            start = half_start + rel_start
            stop = half_start + rel_stop
            init_bias = 0.5 * (1.0 if last_team == "H" else -1.0)

            seg_logit = np.asarray(live_logit[start:stop], dtype=np.float64)
            seg_pb = boundary_prob[start:stop] if boundary_prob is not None else None
            seg_pD = ml_pD[start:stop] if ml_pD is not None else None
            st = viterbi_binary(
                seg_logit, live_sw, init_bias=init_bias,
                boundary_prob=seg_pb, boundary_alpha=b_alpha,
            )
            if min_sp > 1:
                st = absorb_short_runs(
                    st, min_sp,
                    boundary_prob=seg_pb,
                    short_min_len=short_min_sp,
                    boundary_thresh=short_bnd_thr,
                    emission_diff=seg_logit,
                    logit_thresh=short_logit_thr,
                    dead_prob=seg_pD,
                    suffix_dead_thr=suf_thr,
                )

            if pH_hgb is not None and R("use_domino"):
                st = semi_markov_domino_split(
                    st, pH_hgb[start:stop], min_keep=min_sp, min_cand=int(R("domino_min_cand")),
                    prob_thresh=R("domino_prob_thresh"), mean_thresh=R("domino_mean_thresh"),
                )
            if local_logit is not None and R("use_refine"):
                st = refine_boundaries_local(
                    st, local_logit[start:stop], boundary_prob=seg_pb,
                    max_shift=int(R("refine_max_shift")), min_len=int(R("refine_min_len")),
                )

            seg_labels = np.where(st == 1, "H", "A")
            if seg_pD is not None and len(st) > 0:
                breaks = np.flatnonzero(st[1:] != st[:-1]) + 1
                bounds = np.concatenate(([0], breaks, [len(st)]))
                if len(bounds) - 1 >= 2:
                    # 1. Suffix twilight handling right before Dead
                    if suf_len > 0 and stop <= half_end and R("use_suffix"):
                        rs, re = int(bounds[-2]), int(bounds[-1])
                        sign = 1.0 if st[rs] == 1 else -1.0
                        mean_l = float(np.mean(sign * seg_logit[rs:re]))
                        vis_frac = float(np.mean(raw_visible[start + rs : start + re]))
                        mean_pd = float(np.mean(seg_pD[rs:re]))
                        eff_l_thr = R("suffix_long_logit") if (re - rs) >= R("suffix_long_len") else (float(short_logit_thr) if short_logit_thr is not None else 1e9)
                        if (
                            (re - rs) <= suf_len
                            and mean_l < eff_l_thr
                            and (mean_pd >= R("suffix_pd_hi") or (mean_pd >= R("suffix_pd_lo") and vis_frac <= R("suffix_vis_lo")))
                        ):
                            if (re - rs) <= R("suffix_flip_len") and mean_l < R("suffix_flip_logit") and mean_pd < R("suffix_flip_pd"):
                                seg_labels[rs:re] = "A" if st[rs] == 1 else "H"
                            else:
                                seg_labels[rs:re] = "D"

                    # 2. Prefix twilight handling right after Dead
                    if start > half_start and R("use_prefix"):
                        for p_idx in range(len(bounds) - 2):
                            ps, pe = int(bounds[p_idx]), int(bounds[p_idx + 1])
                            psign = 1.0 if st[ps] == 1 else -1.0
                            p_mean_l = float(np.mean(psign * seg_logit[ps:pe]))
                            p_vis_frac = float(np.mean(raw_visible[start + ps : start + pe]))
                            p_mean_pd = float(np.mean(seg_pD[ps:pe]))
                            in_sp = (
                                bool(np.any(sp_restart_windows[start + ps : start + pe]))
                                if sp_restart_windows is not None
                                else False
                            )
                            if (pe - ps) <= R("prefix_max_len") and (
                                p_mean_pd >= R("prefix_pd_hi")
                                or ((pe - ps) <= R("prefix_short_len") and p_vis_frac <= R("prefix_short_vis") and (in_sp or p_mean_pd >= R("prefix_short_pd")))
                                or (p_mean_l < R("prefix_logit_cap") and p_mean_pd >= R("prefix_mid_pd") and p_vis_frac <= R("prefix_mid_vis") and not in_sp)
                                or (
                                    (short_logit_thr is None or p_mean_l < float(short_logit_thr))
                                    and p_mean_pd >= R("prefix_gen_pd")
                                    and (p_vis_frac <= R("prefix_gen_vis") or in_sp)
                                )
                            ):
                                seg_labels[ps:pe] = "D"
                            else:
                                break

            labels[start:stop] = seg_labels
            live_only = seg_labels[seg_labels != "D"]
            if len(live_only):
                last_team = str(live_only[-1])

    return labels


def _ensure_ml_caches():
    """Load `metrica_ml_pH`, `metrica_ml_pD`, `metrica_ml_pBound` (generating via models/ if missing)."""
    ph_path = os.path.join(CACHE_DIR, "metrica_ml_pH.npy")
    pd_path = os.path.join(CACHE_DIR, "metrica_ml_pD.npy")
    pb_path = os.path.join(CACHE_DIR, "metrica_ml_pBound.npy")
    if not (os.path.isfile(ph_path) and os.path.isfile(pd_path) and os.path.isfile(pb_path)):
        print("Generating deterministic ML predictions from saved models/ ...")
        from starter import metrica_possession_model as mpm  # noqa: E402
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], "--predict-only"]
            mpm.main()
        finally:
            sys.argv = old_argv

    return (
        np.load(ph_path).astype(np.float32),
        np.load(pd_path).astype(np.float32),
        np.load(pb_path).astype(np.float32),
    )


def build_algorithmic_baseline(out_dir=RESULTS_DIR, name="Dracula in Barcelona", switch_penalty=None):
    decoder_params = load_decoder_params()
    if switch_penalty is not None:
        decoder_params["live_switch_penalty"] = float(switch_penalty)
    print("Loaded Metrica-calibrated decoder params:", decoder_params)

    print("Loading tactical tracking, aligned Broadcast ATD & Smart Tagging...")
    players, ball, slot_team, events, brd_ball, brd_dist, brd_valid = load_tracking_and_tags()
    n_frames = len(ball)

    print("Cleaning & fusing Tactical + Broadcast ball & computing kinematics...")
    (tac_visible, raw_visible, interp_visible, x_m, y_m, out_of_bounds,
     ball_vx, ball_vy, ball_speed, _) = compute_ball_kinematics(
        ball, brd_ball, brd_valid=brd_valid, players=players, events=events
    )

    print("Computing player proximity (with automatic GK detection) & collective speed...")
    dist_home, dist_away, _, _, player_speed = compute_player_proximity_and_speed(
        players, slot_team, tac_visible, interp_visible, x_m, y_m, ball_vx, ball_vy, brd_dist
    )

    ml_pH, ml_pD, ml_pBound = _ensure_ml_caches()
    ml_pD_raw = ml_pD
    pD_hgb_path = os.path.join(CACHE_DIR, "metrica_ml_pD_hgb.npy")
    pD_seq_path = os.path.join(CACHE_DIR, "metrica_ml_pD_seq.npy")
    pD_bnd_path = os.path.join(CACHE_DIR, "metrica_ml_pD_bnd.npy")
    if os.path.isfile(pD_hgb_path) and os.path.isfile(pD_seq_path) and "dead_seq_weight" in decoder_params:
        w_d = float(decoder_params["dead_seq_weight"])
        ml_pD = (
            w_d * np.load(pD_seq_path).astype(np.float32)
            + (1.0 - w_d) * np.load(pD_hgb_path).astype(np.float32)
        ).astype(np.float32)
    ml_pD_bnd = np.load(pD_bnd_path).astype(np.float32) if os.path.isfile(pD_bnd_path) else None

    dist_min = np.fmin(
        np.where(np.isfinite(dist_home), dist_home, 99.0),
        np.where(np.isfinite(dist_away), dist_away, 99.0),
    )
    gk_rules = _detect_gk_slots_per_half(players, slot_team, GAME)

    print("Stage 1: Decoding Dead Ball (D) via Metrica-calibrated 2-state Viterbi...")
    dead, gk_hold_dir = detect_dead_ball(
        n_frames, tac_visible, raw_visible, out_of_bounds, x_m, y_m,
        ball_speed, player_speed, events, brd_valid, ml_pD=ml_pD, decoder_params=decoder_params,
        ml_pD_bnd=ml_pD_bnd, ball_vx=ball_vx, ball_vy=ball_vy, dist_min=dist_min, gk_rules=gk_rules,
        return_gk_hold=True,
    )

    print("Stage 2: Decoding Live Possession (H/A) via Metrica-calibrated decoder + Smart Tagging prior...")
    tag_prior, bnd_boost, sp_restart_windows = compute_smart_tag_prior(
        n_frames, events, raw_visible, out_of_bounds, ball_speed, ml_pD, ml_pBound=ml_pBound,
    )
    tag_prior = (tag_prior + gk_hold_dir * float(np.log(9.0))).astype(np.float32)
    pH_hgb_path = os.path.join(CACHE_DIR, "metrica_ml_pH_hgb.npy")
    pH_seq_path = os.path.join(CACHE_DIR, "metrica_ml_pH_seq.npy")
    pH_hgb = None
    local_logit = None
    if (
        os.path.isfile(pH_hgb_path) and os.path.isfile(pH_seq_path)
        and "live_seq_weight_vis" in decoder_params
    ):
        from starter.metrica_possession_model import blend_live_logits  # noqa: E402
        pH_hgb = np.load(pH_hgb_path).astype(np.float32)
        pH_seq = np.load(pH_seq_path).astype(np.float32)
        w_vis = float(decoder_params["live_seq_weight_vis"])
        w_miss = float(decoder_params.get("live_seq_weight_miss", 0.5))
        blend_mode = str(decoder_params.get("live_blend_mode", "prob"))
        ml_live_logit = blend_live_logits(
            pH_seq, pH_hgb, raw_visible & (~out_of_bounds),
            w_vis=w_vis, w_miss=w_miss, mode=blend_mode,
        ).astype(np.float32)
        local_logit = (ml_live_logit + tag_prior).astype(np.float32)
        print("  Live gating: w_seq_vis={:.2f}, w_seq_miss={:.2f} ({})".format(w_vis, w_miss, blend_mode))
    else:
        ml_live_logit = logit_from_prob(ml_pH)
    live_logit = ml_live_logit + tag_prior

    labels = decode_live_possessions(
        live_logit, dead, raw_visible, dist_home, dist_away,
        bnd_boost, decoder_params, GAME, ml_pD=ml_pD_raw,
        pH_hgb=pH_hgb, local_logit=local_logit, sp_restart_windows=sp_restart_windows,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}.csv".format(name))
    base_mod.write_submission(out_path, labels)

    counts = {k: int(np.sum(labels == k)) for k in ("H", "A", "D", "X")}
    print("\nWrote {}:".format(os.path.relpath(out_path, ROOT)))
    print("  H: {:>6,}   A: {:>6,}   D: {:>6,}   X: {:>6,}".format(
        counts["H"], counts["A"], counts["D"], counts["X"]
    ))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="Dracula in Barcelona", help="submission filename stem")
    parser.add_argument("--out", default=RESULTS_DIR, help="output directory")
    parser.add_argument("--sw", type=float, default=None, help="override live switch penalty")
    args = parser.parse_args()
    build_algorithmic_baseline(args.out, args.name, switch_penalty=args.sw)
