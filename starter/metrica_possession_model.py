"""Pillar 5: possession, dead-ball & transition-boundary models trained ONLY on Metrica open data.

Rule compliance (RULES.md): no BAR-ATH ground truth / manual annotation / video is used for
training or fitting. All models are trained strictly on github.com/metrica-sports/sample-data
(Sample Games 1-3), whose labels are derived mechanically from Metrica's own event files.
BAR-ATH data is only used (a) at inference time and (b) to measure the *shape* of its
ball-tracking dropouts (visible/missing run lengths inside automatically-detected live play).

Models trained & validated via Leave-One-Game-Out CV on Metrica Games 1-3:
  1. `m_hgb` (HistGradientBoosting): P(Home | live) from ball + player + multi-scale context.
  2. `PossessionSeqNet` (Stage 1 Dilated 1D ResTCN + Bi-GRU, Stage 2 MS-TCN++ Refinement):
       - Head A: P(Home | live) over 1024-frame (41 s) windows with Focal + TMSE loss.
       - Head B: P_boundary(t) = centered probability of a live H <-> A transition [t-3, t+3].
     Blended equally (0.5 * HGB + 0.5 * SeqNet) with team-swap test-time symmetry averaging.
  3. `m_dead` (player-only HistGradientBoosting): P(Dead Ball) using ONLY the player
     constellation & kinematics features (strictly excluding ball-dependent features because
     Metrica nulls the ball during stoppages whereas BAR-ATH often tracks an OOB ball).

Outputs saved to cache/:
  - metrica_ml_pH.npy      : float32 (N,) P(Home | live)
  - metrica_ml_pD.npy      : float32 (N,) player-only P(Dead Ball)
  - metrica_ml_pBound.npy  : float32 (N,) live H <-> A transition boundary probability

Decoder hyperparameters calibrated strictly via LOGO CV on Metrica Games 1-3 are saved to:
  - models/decoder_params.json

Usage:
    python3 starter/metrica_possession_model.py                 # LOGO CV + calibrate decoder + train + predict
    python3 starter/metrica_possession_model.py --predict-only  # deterministic inference from saved models/
    python3 starter/metrica_possession_model.py --no-cv
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from starter.possession_decoder import (  # noqa: E402
    absorb_short_runs,
    logit_from_prob,
    viterbi_binary,
)

METRICA_DIR = os.path.join(ROOT, "external", "sample-data", "data")
CACHE_DIR = os.path.join(ROOT, "cache")
MODEL_DIR = os.path.join(ROOT, "models")
FPS = 25
L, W = 105.0, 68.0

SEQ_LEN = 512
SEQ_STRIDE_TRAIN = 768
SEQ_STRIDE_EVAL = 256

# Prefixes of features that depend on the tracked ball (MUST be excluded from m_dead).
BALL_DEPENDENT_PREFIXES = (
    "bvis", "bx", "by_", "bvx", "bspd", "bacc",
    "dh", "da", "d_", "rbf_", "dv_", "nh", "na", "t_", "last_", "next_",
    "prev_touch_", "next_touch_", "bridge_touch_", "touch_agree", "r_bridge_td", "tc_",
    "rv", "r_ddiff", "r_hclose", "r_bx", "r_bvis", "r_rv", "r_rbf", "r_dv", "r_d_",
)
# Count of tracked players stays ~11 on Metrica but drops with broadcast FOV on BAR-ATH.
NON_DEAD_EXACT = {"h_n", "a_n"}
# Absolute pitch-zone coordinates that leak defensive line depth / pitch position across matches.
ABS_PITCH_EXACT = {
    "h_cx", "a_cx", "h_front", "a_front", "h_back", "a_back", "h_att3", "a_att3",
    "bx", "last_bx", "next_bx", "r_bx6", "r_bx25", "r_bx62", "r_bx125",
}


# ----------------------------------------------------------------------------------------------
# 1. Metrica loading
# ----------------------------------------------------------------------------------------------


def _load_csv_tracking(path):
    df = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(df.columns)
    period = df["Period"].to_numpy()
    frame = df["Frame"].to_numpy()
    xy = []
    for i, c in enumerate(cols):
        if c.startswith("Player") and i + 1 < len(cols):
            xy.append(np.stack([df[c].to_numpy(float), df[cols[i + 1]].to_numpy(float)], axis=1))
    players = np.stack(xy, axis=1)  # (T, P, 2)
    bi = cols.index("Ball")
    ball = np.stack([df["Ball"].to_numpy(float), df[cols[bi + 1]].to_numpy(float)], axis=1)
    return frame, period, players, ball


def _events_csv(path):
    ev = pd.read_csv(path)
    out = []
    for _, r in ev.iterrows():
        out.append(dict(team=str(r["Team"]), type=str(r["Type"]), sub=str(r["Subtype"]).upper(),
                        period=int(r["Period"]), s=int(r["Start Frame"]), e=int(r["End Frame"])))
    return out


def _events_json(path, home_name="Team A"):
    data = json.load(open(path))["data"]
    out = []
    for r in data:
        sub = r.get("subtypes")
        if isinstance(sub, dict):
            sub = sub["name"]
        elif isinstance(sub, list):
            sub = "-".join(x["name"] for x in sub)
        else:
            sub = "NAN"
        team = "Home" if r["team"]["name"] == home_name else "Away"
        out.append(dict(team=team, type=r["type"]["name"], sub=str(sub).upper(), period=int(r["period"]),
                        s=int(r["start"]["frame"]), e=int(r["end"]["frame"] or r["start"]["frame"])))
    return out


def ensure_metrica_sample_data():
    """Ensure public Metrica Sample Data (Games 1-3) exists; clone if absent."""
    if not os.path.isdir(METRICA_DIR):
        print("Metrica Sample Data not found. Auto-cloning github.com/metrica-sports/sample-data ...")
        import subprocess
        ext_dir = os.path.join(ROOT, "external")
        os.makedirs(ext_dir, exist_ok=True)
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/metrica-sports/sample-data.git",
            os.path.join(ext_dir, "sample-data"),
        ])


def load_metrica_game(g):
    ensure_metrica_sample_data()
    d = os.path.join(METRICA_DIR, "Sample_Game_{}".format(g))
    if g in (1, 2):
        fr, per, home, ball = _load_csv_tracking(os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g)))
        _, _, away, _ = _load_csv_tracking(os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(g)))
        events = _events_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(g)))
    else:
        meta = open(os.path.join(d, "Sample_Game_3_metadata.xml")).read()
        h1 = int(re.search(r"first_half_end</Name>\s*<Value>(\d+)", meta).group(1))
        rows = []
        with open(os.path.join(d, "Sample_Game_3_tracking.txt")) as fh:
            for line in fh:
                f, pl, b = line.strip().split(":")
                vals = [float(v) for p in pl.split(";") for v in p.split(",")]
                bx, by = b.split(",")
                rows.append([int(f)] + vals + [float(bx), float(by)])
        arr = np.array(rows, dtype=float)
        fr = arr[:, 0].astype(int)
        pxy = arr[:, 1:45].reshape(len(arr), 22, 2)
        home, away = pxy[:, :11], pxy[:, 11:]
        ball = arr[:, 45:47]
        per = np.where(fr <= h1, 1, 2)
        events = _events_json(os.path.join(d, "Sample_Game_3_events.json"))
    assert np.all(np.diff(fr) == 1)
    off = fr[0]
    for e in events:
        e["s"] -= off
        e["e"] -= off
    to_m = np.array([L, W])
    return dict(home=home * to_m, away=away * to_m, ball=ball * to_m, period=per, events=events)


# ----------------------------------------------------------------------------------------------
# 2. Event-derived labels (0 = Home, 1 = Away, 2 = Dead, -1 = ignore) + Boundaries
# ----------------------------------------------------------------------------------------------

OWN_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def _is_restart(e):
    return e["type"] == "SET PIECE" or "GOAL KICK" in e["sub"]


def event_labels(events, period, T):
    lab = np.full(T, -1, dtype=np.int8)
    evs = sorted([e for e in events if 0 <= e["s"] < T], key=lambda e: (e["s"], e["e"]))
    owner, cur_p, last_f = None, None, 0
    own = [e for e in evs if e["type"] in OWN_TYPES]
    for i, e in enumerate(own):
        if e["period"] != cur_p:
            cur_p, owner, last_f = e["period"], None, e["s"]
        if owner is not None:
            lab[last_f:e["s"]] = owner
        owner = 0 if e["team"] == "Home" else 1
        last_f = e["s"]
        nxt = own[i + 1]["s"] if i + 1 < len(own) and own[i + 1]["period"] == cur_p else min(T, e["e"] + 1)
        lab[e["s"]:nxt] = owner
    dead = np.zeros(T, dtype=bool)
    for i, e in enumerate(evs):
        later = [x for x in evs[i + 1:] if x["period"] == e["period"]]
        stop = None
        if e["type"] == "BALL OUT":
            stop = e["e"]
        elif e["type"] == "SHOT" and ("OUT" in e["sub"] or "GOAL" in e["sub"]):
            stop = e["e"]
        elif e["type"] == "FAULT RECEIVED" or (e["type"] == "CHALLENGE" and "FAULT" in e["sub"]):
            rs = [x for x in later if _is_restart(x)]
            if rs and rs[0]["s"] - e["s"] < 60 * FPS:
                dead[e["s"]:rs[0]["s"]] = True
            continue
        if stop is not None:
            nx = [x for x in later if x["type"] not in ("CARD", "CHALLENGE", "FAULT RECEIVED")]
            if nx:
                dead[stop:nx[0]["s"]] = True
        if _is_restart(e):
            prev = [x for x in evs[:i] if x["period"] == e["period"] and x["type"] not in ("CARD",)]
            if prev:
                pe = max(x["e"] for x in prev[-3:])
                if e["s"] - pe > FPS:
                    dead[pe:e["s"]] = True
    lab[dead] = 2
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        pe = [e for e in evs if e["period"] == p]
        if not pe:
            lab[idx] = -1
            continue
        lab[idx[0]:pe[0]["s"]] = -1
        lab[max(x["e"] for x in pe) + 1:idx[-1] + 1] = -1
    return lab


def challenge_event_labels(lab, events):
    """Apply CHALLENGE.md controlled-possession rule to Metrica event labels.

    Brief single-touch turnovers (< 2 s and < 2 controlled events PASS/CARRY/SET PIECE/SHOT)
    inside a live segment do not count as a possession change.
    """
    out = lab.copy()
    ctrl_events = [
        e for e in events
        if e["type"] in ("PASS", "CARRY", "SET PIECE", "SHOT")
    ]
    live = (out == 0) | (out == 1)
    for ls, le in runs(live):
        changed = True
        while changed:
            changed = False
            seg = out[ls:le]
            breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(seg)]))
            if len(bounds) <= 2:
                break
            # Check shortest interior spell first
            for i in range(1, len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                if (b - a) >= 2 * FPS:
                    continue
                team_str = "Home" if seg[a] == 0 else "Away"
                fa, fb = ls + a, ls + b
                n_ctrl = sum(
                    1 for e in ctrl_events
                    if e["team"] == team_str and e["s"] >= fa and e["s"] < fb
                )
                if n_ctrl < 2:
                    seg[a:b] = seg[a - 1]
                    changed = True
                    break
    return out


def boundary_targets(lab, period, half_win=3):
    """Symmetric binary target centered on direct live H <-> A transition frames [t-3, t+3]."""
    b = np.zeros(len(lab), dtype=np.float32)
    for p in np.unique(period):
        m = np.flatnonzero(period == p)
        lp = lab[m]
        live_idx = np.flatnonzero((lp == 0) | (lp == 1))
        if len(live_idx) < 2:
            continue
        lt = lp[live_idx]
        for idx in np.flatnonzero(lt[1:] != lt[:-1]):
            f1, f2 = live_idx[idx], live_idx[idx + 1]
            if f2 - f1 > 50 or np.any(lp[f1:f2 + 1] == 2):
                continue
            lo = max(0, f2 - half_win)
            hi = min(len(lp), f2 + half_win + 1)
            b[m[lo:hi]] = 1.0
    return b


def dead_boundary_targets(lab, period, half_win=3):
    """Symmetric binary target centered on Live <-> Dead transitions [t-3, t+3]."""
    b = np.zeros(len(lab), dtype=np.float32)
    for p in np.unique(period):
        m = np.flatnonzero(period == p)
        lp = lab[m]
        vi = np.flatnonzero((lp >= 0) & (lp <= 2))
        if len(vi) < 2:
            continue
        is_d = lp[vi] == 2
        for idx in np.flatnonzero(is_d[1:] != is_d[:-1]):
            f1, f2 = vi[idx], vi[idx + 1]
            if f2 - f1 > 50:
                continue
            lo = max(0, f2 - half_win)
            hi = min(len(lp), f2 + half_win + 1)
            b[m[lo:hi]] = 1.0
    return b


# ----------------------------------------------------------------------------------------------
# 3. Direction normalisation + ATD simulation
# ----------------------------------------------------------------------------------------------


def normalise_direction(home, away, ball, period):
    """Flip x per period so the Home team attacks +x (its goalkeeper sits near x = 0)."""
    home, away, ball = home.copy(), away.copy(), ball.copy()
    for p in np.unique(period):
        m = period == p
        with np.errstate(all="ignore"):
            hmin = np.nanmean(np.nanmin(home[m, :, 0], axis=1))
            amin = np.nanmean(np.nanmin(away[m, :, 0], axis=1))
        if hmin > amin:
            for a in (home, away):
                a[m, :, 0] = L - a[m, :, 0]
            ball[m, 0] = L - ball[m, 0]
    return home, away, ball


def runs(mask):
    pad = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(pad)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def simulate_ball_dropout(ball, home, away, live, vis_runs, miss_runs, rng, long_gap_prob=0.30):
    """Hide the ball with BAR-ATH's empirical run lengths mixed with extra 1.5s-5.0s
    flight/occlusion gaps, anchored preferentially to fast ball flight or crowded duels."""
    T = len(ball)
    out = ball.copy()
    spd = np.full(T, 0.0)
    v = np.hypot(*(np.diff(ball, axis=0).T)) * FPS
    spd[1:] = np.nan_to_num(v)
    allp = np.concatenate([home, away], axis=1)
    with np.errstate(all="ignore"):
        dd = np.hypot(allp[:, :, 0] - ball[:, None, 0], allp[:, :, 1] - ball[:, None, 1])
        crowd = np.sum(dd < 2.5, axis=1) >= 3
    trigger = (spd > 8.0) | crowd
    hide = np.zeros(T, dtype=bool)
    for s, e in runs(live):
        t = s + int(rng.integers(0, 30))
        while t < e:
            t += int(rng.choice(vis_runs))
            j = t
            while j < min(e, t + 25) and not trigger[j]:
                j += 1
            t = j
            if t >= e:
                break
            if rng.random() < long_gap_prob:
                g = int(rng.integers(38, 126))
            else:
                g = int(rng.choice(miss_runs))
            hide[t:min(e, t + g)] = True
            t += g
    out[hide] = np.nan
    return out


def simulate_player_noise(arr, rng, sigma=0.14, p_drop=0.08):
    T, P, _ = arr.shape
    out = arr + rng.normal(0, sigma, size=arr.shape)
    for k in range(P):
        n_chunks = int(p_drop * T / 120)
        for _ in range(n_chunks):
            s = int(rng.integers(0, T))
            out[s:s + int(rng.integers(25, 250)), k] = np.nan
    return out


# ----------------------------------------------------------------------------------------------
# 4. Features
# ----------------------------------------------------------------------------------------------


def _roll(x, w):
    """Centred nanmean over window 2*w+1."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    n = cc[hi] - cc[lo]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _ahead(x, w):
    """Forward-looking nanmean over [t, min(T, t+w+1))."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    i = np.arange(len(x))
    lo, hi = i, np.minimum(len(x), i + w + 1)
    n = cc[hi] - cc[lo]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _behind(x, w):
    """Backward-looking nanmean over [max(0, t-w), t+1)."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), i + 1
    n = cc[hi] - cc[lo]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _pass_convergence(px, py, bx, by, bvx, bvy):
    """Team score measuring whether players lie forward along the fast ball flight vector."""
    bv = np.hypot(bvx, bvy)
    fast = np.isfinite(bv) & (bv > 3.5)
    ubx = np.where(fast, bvx / np.maximum(bv, 1e-3), 0.0)[:, None]
    uby = np.where(fast, bvy / np.maximum(bv, 1e-3), 0.0)[:, None]
    dx = px - bx[:, None]
    dy = py - by[:, None]
    dist = np.hypot(dx, dy)
    # Forward projection and perpendicular offset relative to ball trajectory
    proj = dx * ubx + dy * uby
    perp = np.abs(dx * (-uby) + dy * ubx)
    score = np.where(
        fast[:, None] & np.isfinite(dist) & (proj > 0.5) & (proj < 28.0),
        np.exp(-0.5 * (perp / 4.0) ** 2 - 0.5 * (proj / 15.0) ** 2),
        0.0,
    )
    out = np.max(score, axis=1)
    return np.where(fast, out, np.nan)


def _shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _shift_arr(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _ffill_idx(valid):
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _bfill_idx(valid):
    n = len(valid)
    idx = np.where(valid, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _nearest_radial_speed(px, py, pvx, pvy, bx, by, bvx, bvy):
    """Radial velocity of ball relative to the nearest tracked player (>0 = moving away)."""
    T = len(bx)
    dx = bx[:, None] - px
    dy = by[:, None] - py
    dist = np.hypot(dx, dy)
    has_p = np.any(np.isfinite(dist), axis=1)
    idx = np.argmin(np.where(np.isfinite(dist), dist, 1e9), axis=1)
    ar = np.arange(T)
    d0 = np.maximum(dist[ar, idx], 0.1)
    ux = dx[ar, idx] / d0
    uy = dy[ar, idx] / d0
    rv = (bvx - pvx[ar, idx]) * ux + (bvy - pvy[ar, idx]) * uy
    ok = has_p & np.isfinite(bvx) & np.isfinite(pvx[ar, idx])
    return np.where(ok, np.clip(rv, -35.0, 35.0), np.nan)


def _team_dribble_coherence(px, py, pvx, pvy, bx, by, bvx, bvy, sigma_d=3.0, sigma_v=2.0):
    """2D velocity vector error at nearest player and max joint space-velocity Gaussian affinity."""
    T = len(bx)
    dist = np.hypot(bx[:, None] - px, by[:, None] - py)
    has_p = np.any(np.isfinite(dist), axis=1)
    idx = np.argmin(np.where(np.isfinite(dist), dist, 1e9), axis=1)
    ar = np.arange(T)

    dv_all = np.hypot(bvx[:, None] - pvx, bvy[:, None] - pvy)
    dv_near = dv_all[ar, idx]
    ok_near = has_p & np.isfinite(dv_near)
    dv_near = np.where(ok_near, np.clip(dv_near, 0.0, 20.0), np.nan)

    aff = np.exp(-0.5 * (dist / sigma_d) ** 2 - 0.5 * (dv_all / sigma_v) ** 2)
    aff_valid = np.where(np.isfinite(aff), aff, -1.0)
    max_aff = np.max(aff_valid, axis=1)
    max_aff = np.where(max_aff >= 0.0, max_aff, np.nan)
    return dv_near, max_aff


TOUCH_FEATURES = os.environ.get("POSSESSION_TOUCH_FEATURES") == "1"


def touch_event_features(bvis, d_home, d_away, acc, radius=3.0, acc_min=4.0, cap=150):
    """Candidate-touch features in the spirit of ELASTIC (Kim et al.): a candidate touch by a team
    is a tracked-ball frame where that team's nearest player is within `radius` metres, closer
    than the opponent's, and either at a local minimum of that distance or at a local maximum
    of ball acceleration (>= `acc_min`).  Each candidate gets a "kick distance": how far the ball
    then separates from that team before the next candidate (clipped at `radius`), which tells a
    controlled touch that keeps the ball close from a kick, a deflection or a fly-by.

    All names start with `tc_` so the player-only dead-ball model never sees them.
    """
    T = len(bvis)
    dh = np.where(np.isfinite(d_home), d_home, 99.0)
    da = np.where(np.isfinite(d_away), d_away, 99.0)
    a = np.where(np.isfinite(acc), acc, 0.0)
    accpk = np.zeros(T, dtype=bool)
    accpk[1:-1] = (a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] >= acc_min)
    cands = []
    for d, o in ((dh, da), (da, dh)):
        locmin = np.zeros(T, dtype=bool)
        locmin[2:-2] = (
            (d[2:-2] <= d[1:-3]) & (d[2:-2] <= d[3:-1]) & (d[2:-2] <= d[:-4]) & (d[2:-2] <= d[4:])
        )
        cands.append(bvis & (d <= radius) & (d < o) & (locmin | accpk))
    any_c = cands[0] | cands[1]
    idx = np.arange(T)
    nxt_any = np.minimum.accumulate(np.where(any_c, idx, T)[::-1])[::-1]
    nxt_after = np.concatenate((nxt_any[1:], [T]))  # first candidate strictly after t

    F = {}
    last_frame = []
    for name, c, d in (("h", cands[0], dh), ("a", cands[1], da)):
        # kick distance of every candidate: max separation from this team until the next candidate
        kd_at = np.full(T, np.nan)
        dfin = np.where(bvis, np.minimum(d, radius), -1.0)
        for t in np.flatnonzero(c):
            e = min(int(nxt_after[t]), t + cap, T)
            seg = dfin[t + 1:e]
            if len(seg) and seg.max() >= 0.0:
                kd_at[t] = seg.max()
        fi = np.maximum.accumulate(np.where(c, idx, -1))
        bi = np.minimum.accumulate(np.where(c, idx, T)[::-1])[::-1]
        last_frame.append(fi)
        F["tc_since_" + name] = np.where(fi >= 0, np.minimum(idx - fi, cap), cap).astype(float)
        F["tc_until_" + name] = np.where(bi < T, np.minimum(bi - idx, cap), cap).astype(float)
        F["tc_kd_last_" + name] = np.where(fi >= 0, kd_at[np.clip(fi, 0, T - 1)], np.nan)
        cs = np.concatenate(([0.0], np.cumsum(c.astype(float))))
        F["tc_cnt50_" + name] = cs[np.minimum(T, idx + 51)] - cs[np.maximum(0, idx - 50)]
    F["tc_since_diff"] = F["tc_since_a"] - F["tc_since_h"]
    F["tc_until_diff"] = F["tc_until_a"] - F["tc_until_h"]
    F["tc_cnt50_diff"] = F["tc_cnt50_h"] - F["tc_cnt50_a"]
    # who touched last, and how controlled that touch was (+ home, - away)
    lh, la = last_frame
    last_team = np.where((lh < 0) & (la < 0), 0.0, np.where(lh >= la, 1.0, -1.0))
    kd_last = np.where(last_team > 0, F["tc_kd_last_h"], F["tc_kd_last_a"])
    F["tc_last_team"] = last_team
    F["tc_last_ctrl"] = last_team * (1.0 - np.where(np.isfinite(kd_last), kd_last, radius) / radius)
    return F


def segment_features(home, away, ball):
    """home/away: (T, P, 2) metres (NaN allowed), Home attacks +x. ball: (T, 2) with NaN."""
    T = len(ball)
    F = {}
    with np.errstate(all="ignore"):
        hx, hy, ax, ay = home[:, :, 0], home[:, :, 1], away[:, :, 0], away[:, :, 1]
        # outfield: drop the deepest player of each team (goalkeeper)
        hx_o = np.where(hx == np.nanmin(hx, axis=1, keepdims=True), np.nan, hx)
        ax_o = np.where(ax == np.nanmax(ax, axis=1, keepdims=True), np.nan, ax)
        hy_o = np.where(np.isfinite(hx_o), hy, np.nan)
        ay_o = np.where(np.isfinite(ax_o), ay, np.nan)
        F["h_cx"] = np.nanmean(hx_o, 1)
        F["a_cx"] = np.nanmean(ax_o, 1)
        F["h_sx"] = np.nanstd(hx_o, 1)
        F["a_sx"] = np.nanstd(ax_o, 1)
        F["h_sy"] = np.nanstd(hy_o, 1)
        F["a_sy"] = np.nanstd(ay_o, 1)
        F["h_front"] = np.nanmax(hx_o, 1)
        F["a_front"] = np.nanmin(ax_o, 1)
        F["h_back"] = np.nanmin(hx_o, 1)
        F["a_back"] = np.nanmax(ax_o, 1)
        F["cx_diff"] = F["h_cx"] - F["a_cx"]
        F["sy_diff"] = F["h_sy"] - F["a_sy"]
        F["sx_diff"] = F["h_sx"] - F["a_sx"]
        F["h_att3"] = np.sum(hx_o > 70, 1).astype(float)
        F["a_att3"] = np.sum(ax_o < 35, 1).astype(float)
        F["h_n"] = np.sum(np.isfinite(hx), 1).astype(float)
        F["a_n"] = np.sum(np.isfinite(ax), 1).astype(float)

        def pvel(a):
            d = (_shift_arr(a, 5) - _shift_arr(a, -5)) / (10 / FPS)
            s = np.hypot(d[..., 0], d[..., 1])
            bad = s > 12.0
            vx = np.where(bad, np.nan, d[..., 0])
            vy = np.where(bad, np.nan, d[..., 1])
            s = np.where(bad, np.nan, s)
            return vx, vy, s

        hvx, hvy, hs = pvel(home)
        avx, avy, as_ = pvel(away)
        all_s = np.concatenate([hs, as_], axis=1)
        F["h_spd"] = np.nanmean(hs, 1)
        F["a_spd"] = np.nanmean(as_, 1)
        F["all_spd_mean"] = np.nanmean(all_s, 1)
        F["all_spd_std"] = np.nanstd(all_s, 1)
        F["all_p50"] = np.nanmedian(all_s, 1)
        F["all_p90"] = np.nanpercentile(all_s, 90, axis=1)
        fast = np.where(np.isfinite(all_s), (all_s > 3.0).astype(float), np.nan)
        sprint = np.where(np.isfinite(all_s), (all_s > 4.8).astype(float), np.nan)
        walk = np.where(np.isfinite(all_s), (all_s < 1.5).astype(float), np.nan)
        F["all_fast"] = np.nanmean(fast, 1)
        F["all_sprint"] = np.nanmean(sprint, 1)
        F["all_walk"] = np.nanmean(walk, 1)
        F["spd_diff"] = F["h_spd"] - F["a_spd"]
        F["cx_sep"] = np.abs(F["a_cx"] - F["h_cx"])
        F["all_spread_x"] = F["h_sx"] + F["a_sx"]
        F["all_spread_y"] = F["h_sy"] + F["a_sy"]

        # Local extreme speeds (top-1 / top-3 fastest players) to avoid 22-player mean dilution
        s_fill = np.where(np.isfinite(all_s), all_s, -1.0)
        s_sorted = np.sort(s_fill, axis=1)
        top1 = s_sorted[:, -1]
        top3 = s_sorted[:, -3:]
        t3_ok = top3 >= 0.0
        F["top1_spd"] = np.where(top1 >= 0.0, top1, np.nan)
        F["top3_spd"] = np.where(
            np.any(t3_ok, axis=1),
            np.sum(np.where(t3_ok, top3, 0.0), axis=1) / np.maximum(1, np.sum(t3_ok, axis=1)),
            np.nan,
        )

        # Min distance between actively moving opponents (speed > 1.2 m/s): pressing / live duel
        opp_press = np.full(T, np.nan, dtype=np.float32)
        chunk = 4000
        for i0 in range(0, T, chunk):
            i1 = min(T, i0 + chunk)
            mh = np.isfinite(hs[i0:i1]) & (hs[i0:i1] > 1.2) & np.isfinite(hx[i0:i1]) & np.isfinite(hy[i0:i1])
            ma = np.isfinite(as_[i0:i1]) & (as_[i0:i1] > 1.2) & np.isfinite(ax[i0:i1]) & np.isfinite(ay[i0:i1])
            dx = hx[i0:i1, :, None] - ax[i0:i1, None, :]
            dy = hy[i0:i1, :, None] - ay[i0:i1, None, :]
            d_ha = np.hypot(dx, dy)
            pair_ok = mh[:, :, None] & ma[:, None, :] & np.isfinite(d_ha)
            dmin = np.min(np.where(pair_ok, d_ha, 999.0), axis=(1, 2))
            opp_press[i0:i1] = np.where(dmin < 900.0, dmin, np.nan)
        F["opp_press_min"] = opp_press

    F["all_spd_r25"] = _roll(F["all_spd_mean"], 25)
    F["all_spd_r62"] = _roll(F["all_spd_mean"], 62)
    F["all_spd_dsp"] = _roll(F["all_spd_mean"], 12) - _roll(F["all_spd_mean"], 62)
    spd_sm = _roll(F["all_spd_mean"], 5)
    F["all_spd_fwd"] = _shift(spd_sm, 25) - spd_sm
    F["all_spd_bwd"] = spd_sm - _shift(spd_sm, -25)
    F["all_spd_dv"] = F["all_spd_fwd"] - F["all_spd_bwd"]
    F["r_all_p50_25"] = _roll(F["all_p50"], 25)
    F["r_all_p90_25"] = _roll(F["all_p90"], 25)
    F["r_all_fast_25"] = _roll(F["all_fast"], 25)
    F["all_sprint_r25"] = _roll(F["all_sprint"], 25)
    F["all_walk_r25"] = _roll(F["all_walk"], 25)
    F["r_top1_spd25"] = _roll(F["top1_spd"], 25)
    F["r_top3_spd25"] = _roll(F["top3_spd"], 25)
    F["r_top3_spd62"] = _roll(F["top3_spd"], 62)
    F["r_opp_press25"] = _roll(F["opp_press_min"], 25)
    F["r_opp_press62"] = _roll(F["opp_press_min"], 62)

    hc, ac = _roll(F["h_cx"], 3), _roll(F["a_cx"], 3)
    for k in (5, 12, 25, 50):
        F["h_vx{}".format(k)] = (_shift(hc, k) - _shift(hc, -k)) / (2 * k / FPS)
        F["a_vx{}".format(k)] = (_shift(ac, k) - _shift(ac, -k)) / (2 * k / FPS)
    for name, c in (("h", hc), ("a", ac)):
        vf = (_shift(c, 25) - c)
        vb = (c - _shift(c, -25))
        F[name + "_vfwd"], F[name + "_vbwd"], F[name + "_dv"] = vf, vb, vf - vb
    sy = _roll(F["sy_diff"], 3)
    F["sy_dv"] = (_shift(sy, 25) - sy) - (sy - _shift(sy, -25))

    # ---- ball ----
    bvis = np.isfinite(ball[:, 0])
    bx, by = ball[:, 0], ball[:, 1]
    F["bvis"] = bvis.astype(float)
    F["bx"] = bx
    F["by_c"] = np.abs(by - W / 2)
    with np.errstate(all="ignore"):
        bvx = np.full(T, np.nan)
        bvy = np.full(T, np.nan)
        bvx[1:-1] = (bx[2:] - bx[:-2]) * FPS / 2
        bvy[1:-1] = (by[2:] - by[:-2]) * FPS / 2
        bv = np.hypot(bvx, bvy)
        F["bvx"] = bvx
        F["bspd"] = bv
        acc = np.full(T, np.nan)
        acc[1:-1] = np.abs(bv[2:] - bv[:-2]) * FPS / 2
        F["bacc"] = acc
        dh = np.sort(np.hypot(hx - bx[:, None], hy - by[:, None]), axis=1)
        da = np.sort(np.hypot(ax - bx[:, None], ay - by[:, None]), axis=1)
        F["rvh"] = _nearest_radial_speed(hx, hy, hvx, hvy, bx, by, bvx, bvy)
        F["rva"] = _nearest_radial_speed(ax, ay, avx, avy, bx, by, bvx, bvy)
        F["rv_diff"] = np.clip(F["rva"] - F["rvh"], -30.0, 30.0)
        F["dv_h"], F["dv_match_h"] = _team_dribble_coherence(hx, hy, hvx, hvy, bx, by, bvx, bvy)
        F["dv_a"], F["dv_match_a"] = _team_dribble_coherence(ax, ay, avx, avy, bx, by, bvx, bvy)
        F["dv_diff"] = np.clip(F["dv_a"] - F["dv_h"], -20.0, 20.0)
        F["dv_match_diff"] = F["dv_match_h"] - F["dv_match_a"]
    for k in range(3):
        F["dh{}".format(k + 1)] = dh[:, k]
        F["da{}".format(k + 1)] = da[:, k]
    diff = np.clip(da[:, 0] - dh[:, 0], -15, 15)
    F["d_diff"] = diff
    with np.errstate(all="ignore"):
        F["d_logr"] = np.log((da[:, 0] + 0.3) / (dh[:, 0] + 0.3))
        F["nh5"] = np.where(bvis, np.sum(np.hypot(hx - bx[:, None], hy - by[:, None]) < 5, 1), np.nan)
        F["na5"] = np.where(bvis, np.sum(np.hypot(ax - bx[:, None], ay - by[:, None]) < 5, 1), np.nan)
        # Soft Gaussian RBF proximity (sigma = 2.5 m) and tight touch RBF (sigma = 1.2 m)
        rbf_h = np.where(bvis[:, None], np.exp(-0.5 * (dh[:, :3] / 2.5) ** 2), np.nan)
        rbf_a = np.where(bvis[:, None], np.exp(-0.5 * (da[:, :3] / 2.5) ** 2), np.nan)
        F["rbf_h"] = np.where(bvis, np.nansum(rbf_h, axis=1), np.nan)
        F["rbf_a"] = np.where(bvis, np.nansum(rbf_a, axis=1), np.nan)
        F["rbf_diff"] = F["rbf_h"] - F["rbf_a"]
        F["rbf_logr"] = np.log((F["rbf_h"] + 0.05) / (F["rbf_a"] + 0.05))
        rbf_12_h = np.where(bvis[:, None], np.exp(-0.5 * (dh[:, :3] / 1.2) ** 2), np.nan)
        rbf_12_a = np.where(bvis[:, None], np.exp(-0.5 * (da[:, :3] / 1.2) ** 2), np.nan)
        F["rbf_12_h"] = np.where(bvis, np.nansum(rbf_12_h, axis=1), np.nan)
        F["rbf_12_a"] = np.where(bvis, np.nansum(rbf_12_a, axis=1), np.nan)
        F["rbf_12_diff"] = F["rbf_12_h"] - F["rbf_12_a"]
        F["d_ctrl_margin"] = np.where(bvis, np.exp(-dh[:, 0] / 1.5) - np.exp(-da[:, 0] / 1.5), np.nan)
        F["d_pconv_h"] = _pass_convergence(hx, hy, bx, by, bvx, bvy)
        F["d_pconv_a"] = _pass_convergence(ax, ay, bx, by, bvx, bvy)
        F["d_pconv_diff"] = F["d_pconv_h"] - F["d_pconv_a"]

    fi, bi = _ffill_idx(bvis), _bfill_idx(bvis)
    i = np.arange(T)
    F["t_since"] = np.where(fi >= 0, np.minimum(i - fi, 300), 300).astype(float)
    F["t_until"] = np.where(bi < T, np.minimum(bi - i, 300), 300).astype(float)
    fic, bic = np.clip(fi, 0, T - 1), np.clip(bi, 0, T - 1)
    for nm, arr in (("d_diff", diff), ("bx", bx), ("bspd", F["bspd"]), ("rv_diff", F["rv_diff"])):
        F["last_" + nm] = np.where(fi >= 0, arr[fic], np.nan)
        F["next_" + nm] = np.where(bi < T, arr[bic], np.nan)

    hcloser = np.where(bvis, (diff > 0).astype(float), np.nan)
    for w in (6, 25, 62, 125):
        F["r_ddiff{}".format(w)] = _roll(diff, w)
        F["r_hclose{}".format(w)] = _roll(hcloser, w)
        F["r_bx{}".format(w)] = _roll(bx, w)
        F["r_bvis{}".format(w)] = _roll(F["bvis"], w)
    # Forward/backward asymmetry features so transition boundaries stay crisp
    F["d_lead12"] = _ahead(diff, 12)
    F["d_lag12"] = _behind(diff, 12)
    F["d_asym12"] = F["d_lead12"] - F["d_lag12"]
    F["d_lead25"] = _ahead(diff, 25)
    F["d_lag25"] = _behind(diff, 25)
    F["d_asym25"] = F["d_lead25"] - F["d_lag25"]
    F["rbf_lead12"] = _ahead(F["rbf_diff"], 12)
    F["rbf_lag12"] = _behind(F["rbf_diff"], 12)
    F["r_d_pconv6"] = _roll(F["d_pconv_diff"], 6)
    F["r_rbf_diff25"] = _roll(F["rbf_diff"], 25)
    F["r_rbf_diff62"] = _roll(F["rbf_diff"], 62)
    F["r_dv_match12"] = _roll(F["dv_match_diff"], 12)
    F["r_dv_match25"] = _roll(F["dv_match_diff"], 25)
    for nm in ("rvh", "rva", "rv_diff"):
        F["r_{}6".format(nm)] = _roll(F[nm], 6)
    for w in (25, 62):
        for nm in ("cx_diff", "sy_diff", "spd_diff", "h_vx12", "a_vx12"):
            F["r_{}{}".format(nm, w)] = _roll(F[nm], w)
    if TOUCH_FEATURES:
        F.update(touch_event_features(bvis, dh[:, 0], da[:, 0], acc))
    names = sorted(F)
    return np.stack([F[n].astype(np.float32) for n in names], axis=1), names


def features_by_period(home, away, ball, period):
    X = None
    names = None
    for p in np.unique(period):
        m = np.flatnonzero(period == p)
        Xp, names = segment_features(home[m], away[m], ball[m])
        if X is None:
            X = np.full((len(period), Xp.shape[1]), np.nan, dtype=np.float32)
        X[m] = Xp
    return X, names


def player_only_indices(names):
    idx = [
        i for i, n in enumerate(names)
        if not n.startswith(BALL_DEPENDENT_PREFIXES)
        and n not in NON_DEAD_EXACT
        and n not in ABS_PITCH_EXACT
    ]
    pnames = [names[i] for i in idx]
    assert "spd_diff" in pnames and "r_spd_diff25" in pnames
    assert "top3_spd" in pnames and "opp_press_min" in pnames
    assert not any(n in pnames for n in ("bvis", "d_diff", "bspd", "rvh", "bx", "h_n", "a_n", "h_cx", "a_cx"))
    return idx


def seq_live_indices(names):
    """Pitch-translation-invariant feature indices for PossessionSeqNet cross-domain generalization."""
    idx = [
        i for i, n in enumerate(names)
        if n not in ABS_PITCH_EXACT and n not in NON_DEAD_EXACT
    ]
    snames = [names[i] for i in idx]
    assert "d_diff" in snames and "rbf_diff" in snames and "cx_diff" in snames
    assert not any(n in snames for n in ("bx", "h_cx", "a_cx", "h_front", "r_bx25", "h_n", "a_n"))
    return idx


# ----------------------------------------------------------------------------------------------
# 5. BAR-ATH side
# ----------------------------------------------------------------------------------------------


def load_bar_ath():
    from starter import algorithmic_baseline as ab
    players, ball, slot_team, events, brd_ball, brd_dist, brd_valid = ab.load_tracking_and_tags()
    (tac_vis, raw_vis, interp_vis, x_m, y_m, oob, bvx, bvy, bspd, bacc) = ab.compute_ball_kinematics(
        ball, brd_ball, brd_valid=brd_valid, players=players, events=events
    )
    dh, da, rvh, rva, pspd = ab.compute_player_proximity_and_speed(
        players, slot_team, tac_vis, interp_vis, x_m, y_m, bvx, bvy, brd_dist)
    dead = ab.detect_dead_ball(len(ball), tac_vis, raw_vis, oob, x_m, y_m, bspd, pspd, events, brd_valid)
    P = np.asarray(players[:, :, :2], dtype=np.float64) * np.array([L, W])
    home = P[:, slot_team == 0]
    away = P[:, slot_team == 1]
    bxy = np.stack([x_m, y_m], 1)
    bxy[~raw_vis] = np.nan
    period = np.where(np.arange(len(ball)) < half_video_blocks(GAME)[1][1], 1, 2)
    return home, away, bxy, period, ~dead, raw_vis & ~oob


def bar_ath_run_lengths(live, vis):
    vr, mr = [], []
    for s, e in runs(live):
        if e - s < 50:
            continue
        v = vis[s:e]
        vr += [b - a for a, b in runs(v)]
        mr += [b - a for a, b in runs(~v)]
    return np.array(vr), np.array(mr)


# ----------------------------------------------------------------------------------------------
# 6. Training data construction (returns full period sequences + labels for both GBDT & SeqNet)
# ----------------------------------------------------------------------------------------------


def build_game_copies(game, vis_runs, miss_runs, seed, n_copies=8):
    """Build n_copies ATD-corrupted + 4-fold football-symmetry sequence dicts per period."""
    home, away, ball = normalise_direction(game["home"], game["away"], game["ball"], game["period"])
    T = len(ball)
    lab = event_labels(game["events"], game["period"], T)
    lab_ch = challenge_event_labels(lab, game["events"])
    bnd = boundary_targets(lab, game["period"], half_win=3)
    d_bnd = dead_boundary_targets(lab, game["period"], half_win=3)
    live_track = (lab >= 0) & (lab <= 1) & np.isfinite(ball[:, 0])
    # Restrict labelled Dead frames to valid periods (>=25 f stoppages so tiny 1-frame event gaps stay sharp)
    valid_dead = (lab == 2)
    rng = np.random.default_rng(seed)
    seqs = []
    names = None
    for c in range(n_copies):
        swap_x = (c % 2 == 1)
        flip_y = ((c // 2) % 2 == 1)
        if not swap_x:
            h, a, b, y, y_ch = home.copy(), away.copy(), ball.copy(), lab, lab_ch
        else:
            h = away.copy(); a = home.copy(); b = ball.copy()
            h[..., 0] = L - h[..., 0]; a[..., 0] = L - a[..., 0]; b[:, 0] = L - b[:, 0]
            y = np.where(lab == 0, 1, np.where(lab == 1, 0, lab))
            y_ch = np.where(lab_ch == 0, 1, np.where(lab_ch == 1, 0, lab_ch))
        if flip_y:
            h[..., 1] = W - h[..., 1]
            a[..., 1] = W - a[..., 1]
            b[:, 1] = W - b[:, 1]
        long_p = 0.08 if c < 2 else 0.12
        bm = simulate_ball_dropout(b, h, a, live_track, vis_runs, miss_runs, rng, long_gap_prob=long_p)
        # Simulate homography parallax wobble (~0.56 m std on eval copies, ~1.05 m + bursts on c>=2)
        if c < 2:
            bm[:, 0] += _roll(rng.normal(0.0, 2.0, size=T), 12)
            bm[:, 1] += _roll(rng.normal(0.0, 2.0, size=T), 12)
        else:
            bm[:, 0] += _roll(rng.normal(0.0, 4.5, size=T), 18)
            bm[:, 1] += _roll(rng.normal(0.0, 4.5, size=T), 18)
            for _ in range(int(0.08 * T / 40)):
                ps = int(rng.integers(0, T))
                pe = min(T, ps + int(rng.integers(15, 50)))
                bm[ps:pe, 0] += rng.normal(0.0, 1.8)
                bm[ps:pe, 1] += rng.normal(0.0, 1.8)
        bm[~live_track] = np.nan
        hp = simulate_player_noise(h, rng)
        ap = simulate_player_noise(a, rng)
        X, names = features_by_period(hp, ap, bm, game["period"])
        for p in np.unique(game["period"]):
            m = np.flatnonzero(game["period"] == p)
            seqs.append(dict(
                X=X[m],
                y=y[m],
                y_ch=y_ch[m],
                live=live_track[m],
                dead=valid_dead[m],
                bnd=bnd[m],
                d_bnd=d_bnd[m],
                copy=c,
            ))
    return seqs, names


def _make_dead_seq_view(seq_list, p_idx):
    """Project period sequences onto strictly player-only features with Dead-vs-Live targets."""
    return [
        dict(
            X=s["X"][:, p_idx],
            y=(s["y"] == 2).astype(np.int8),
            y_ch=(s["y"] == 2).astype(np.int8),
            live=(s["y"] >= 0) & (s["y"] <= 2),
            dead=s["dead"],
            bnd=s["d_bnd"],
            copy=s["copy"],
        )
        for s in seq_list
    ]


def _make_live_seq_view(seq_list, seq_idx):
    """Project period sequences onto pitch-invariant live H/A features for PossessionSeqNet."""
    return [
        dict(
            X=s["X"][:, seq_idx],
            y=s["y"],
            y_ch=s["y_ch"],
            live=s["live"],
            dead=s["dead"],
            bnd=s["bnd"],
            copy=s["copy"],
        )
        for s in seq_list
    ]


# ----------------------------------------------------------------------------------------------
# 7. Tabular HGB models (live H/A + player-only Dead Ball)
# ----------------------------------------------------------------------------------------------


def make_hgb(max_iter=450, min_samples_leaf=200, max_leaf_nodes=63, random_state=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=random_state)


def collect_tabular_live(seq_list, step=2):
    Xs, ys = [], []
    for s in seq_list:
        keep = np.flatnonzero(s["live"])[::step]
        if len(keep):
            Xs.append(s["X"][keep])
            ys.append(s["y"][keep])
    return np.concatenate(Xs), np.concatenate(ys)


def collect_tabular_dead(seq_list, player_idx, step=3):
    Xs, ys = [], []
    for s in seq_list:
        mask = s["live"] | s["dead"]
        keep = np.flatnonzero(mask)[::step]
        if len(keep):
            Xs.append(s["X"][keep][:, player_idx])
            ys.append((s["y"][keep] == 2).astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys)


# ----------------------------------------------------------------------------------------------
# 8. PyTorch Sequence Model: Dilated 1D ResTCN + 2-layer Bi-GRU
# ----------------------------------------------------------------------------------------------


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def set_deterministic(seed=2027):
    """Enable deterministic PyTorch/NumPy execution for reproducible training and inference."""
    import random
    torch, _ = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(torch, device_choice="cpu"):
    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "mps" or (device_choice == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def build_seq_net(in_dim, hidden=96, arch="bigru"):
    torch, nn = _import_torch()

    class ResTCNBlock(nn.Module):
        def __init__(self, channels, dilation, dropout=0.25):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            self.gn1 = nn.GroupNorm(8, channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            self.gn2 = nn.GroupNorm(8, channels)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.act(self.gn1(self.conv1(x)))
            h = self.drop(h)
            h = self.gn2(self.conv2(h))
            return self.act(x + h)

    class PossessionSeqNet(nn.Module):
        def __init__(self, c_in, h=hidden):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(c_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            self.tcn = nn.Sequential(
                ResTCNBlock(h, dilation=1, dropout=0.25),
                ResTCNBlock(h, dilation=2, dropout=0.25),
                ResTCNBlock(h, dilation=4, dropout=0.25),
                ResTCNBlock(h, dilation=8, dropout=0.25),
                ResTCNBlock(h, dilation=16, dropout=0.25),
            )
            self.gru = nn.GRU(
                h, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.30
            )
            self.aux_poss = nn.Linear(128, 1)
            self.aux_bound = nn.Linear(128, 1)
            self.refine_tcn = nn.Sequential(
                ResTCNBlock(128, dilation=2, dropout=0.25),
                ResTCNBlock(128, dilation=4, dropout=0.25),
                ResTCNBlock(128, dilation=8, dropout=0.25),
            )
            self.head_poss = nn.Linear(128, 1)
            self.head_bound = nn.Linear(128, 1)

        def forward(self, x):
            z = self.proj(x)              # (B, T, h)
            z = self.tcn(z.transpose(1, 2)).transpose(1, 2)
            out, _ = self.gru(z)          # (B, T, 128)
            r = self.refine_tcn(out.transpose(1, 2)).transpose(1, 2)
            lp2 = self.head_poss(r).squeeze(-1)
            lb2 = self.head_bound(r).squeeze(-1)
            if self.training:
                lp1 = self.aux_poss(out).squeeze(-1)
                lb1 = self.aux_bound(out).squeeze(-1)
                return lp2, lb2, lp1, lb1
            return lp2, lb2

    class DeepResTCN(nn.Module):
        """Pure multi-scale dilated residual TCN with dual-stage boundary refinement."""

        def __init__(self, c_in, h=128):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(c_in, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            self.stem = nn.Sequential(
                ResTCNBlock(h, dilation=1, dropout=0.20),
                ResTCNBlock(h, dilation=2, dropout=0.20),
                ResTCNBlock(h, dilation=4, dropout=0.20),
                ResTCNBlock(h, dilation=8, dropout=0.20),
                ResTCNBlock(h, dilation=16, dropout=0.20),
                ResTCNBlock(h, dilation=32, dropout=0.20),
            )
            self.aux_poss = nn.Linear(h, 1)
            self.aux_bound = nn.Linear(h, 1)
            self.refine_tcn = nn.Sequential(
                ResTCNBlock(h, dilation=1, dropout=0.20),
                ResTCNBlock(h, dilation=2, dropout=0.20),
                ResTCNBlock(h, dilation=4, dropout=0.20),
                ResTCNBlock(h, dilation=8, dropout=0.20),
            )
            self.head_poss = nn.Linear(h, 1)
            self.head_bound = nn.Linear(h, 1)

        def forward(self, x):
            z = self.proj(x)
            z = self.stem(z.transpose(1, 2)).transpose(1, 2)
            r = self.refine_tcn(z.transpose(1, 2)).transpose(1, 2)
            lp2 = self.head_poss(r).squeeze(-1)
            lb2 = self.head_bound(r).squeeze(-1)
            if self.training:
                lp1 = self.aux_poss(z).squeeze(-1)
                lb1 = self.aux_bound(z).squeeze(-1)
                return lp2, lb2, lp1, lb1
            return lp2, lb2

    if arch == "tcn":
        return DeepResTCN(in_dim, h=128 if hidden == 96 else hidden)
    return PossessionSeqNet(in_dim, hidden)


def fit_scaler(seq_list):
    """Compute per-feature mean & std on finite values across sequences."""
    c = seq_list[0]["X"].shape[1]
    cnt = np.zeros(c, dtype=np.float64)
    s1 = np.zeros(c, dtype=np.float64)
    s2 = np.zeros(c, dtype=np.float64)
    for s in seq_list:
        X = s["X"][::2]
        m = np.isfinite(X)
        Xv = np.where(m, X, 0.0)
        cnt += m.sum(axis=0)
        s1 += Xv.sum(axis=0)
        s2 += (Xv * Xv).sum(axis=0)
    mean = (s1 / np.maximum(cnt, 1.0)).astype(np.float32)
    var = np.maximum(s2 / np.maximum(cnt, 1.0) - mean.astype(np.float64) ** 2, 1e-4)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def transform_features(X, mean, std):
    """Standardise X, fill NaNs to 0, and append binary finiteness mask."""
    fin = np.isfinite(X)
    Xn = np.where(fin, (X - mean[None, :]) / std[None, :], 0.0).astype(np.float32)
    Xn = np.clip(Xn, -8.0, 8.0)
    return np.concatenate([Xn, fin.astype(np.float32)], axis=1)


def _window_slices(T, win=SEQ_LEN, stride=SEQ_STRIDE_TRAIN, offset=0):
    if T <= win:
        return [(0, T)]
    start0 = min(max(0, int(offset)), max(0, T - win))
    starts = list(range(start0, T - win + 1, stride))
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    if starts[-1] != T - win:
        starts.append(T - win)
    return [(s, s + win) for s in starts]


def train_seq_net(train_seqs, mean, std, epochs=12, batch_size=32, lr=1.5e-3, seed=42, device_choice="cpu", arch="bigru"):
    torch, nn = _import_torch()
    set_deterministic(seed)
    np_rng = np.random.default_rng(seed)
    device = get_device(torch, device_choice)

    windows = []
    pos_b, tot_b = 0.0, 0.0
    for s in train_seqs:
        Xt = s["X"]  # raw view; standardised per batch below (same numbers, a third of the memory)
        live = s["live"]
        y = s["y"]
        bnd = s["bnd"]
        T = len(y)
        phase = (int(s.get("copy", 0)) * 96) % max(1, SEQ_STRIDE_TRAIN)
        for a, b in _window_slices(T, SEQ_LEN, SEQ_STRIDE_TRAIN, offset=phase):
            if b - a < 128 or live[a:b].sum() < 48:
                continue
            X_w = Xt[a:b]
            if len(X_w) < SEQ_LEN:
                continue
            lw = live[a:b]
            yw = np.where(y[a:b] == 1, 1.0, 0.0).astype(np.float32)
            bw = bnd[a:b].astype(np.float32)
            windows.append((X_w, yw, bw, lw.astype(np.float32)))
            pos_b += float(bw[lw].sum())
            tot_b += float(lw.sum())

    # Bounded positive weight so P_boundary separates switch windows cleanly without saturating
    raw_pw = (tot_b - pos_b) / max(pos_b, 1.0)
    bnd_pos_weight = float(np.clip(np.sqrt(raw_pw), 2.5, 6.0))

    c_in = 2 * windows[0][0].shape[1]
    model = build_seq_net(c_in, arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    bce_poss = nn.BCEWithLogitsLoss(reduction="none")
    bce_bnd = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([bnd_pos_weight], device=device),
    )

    idx = np.arange(len(windows))
    model.train()
    for ep in range(epochs):
        np_rng.shuffle(idx)
        for i0 in range(0, len(idx), batch_size):
            batch_idx = idx[i0:i0 + batch_size]
            xb = torch.from_numpy(
                np.stack([transform_features(windows[j][0], mean, std) for j in batch_idx])).to(device)
            yb = torch.from_numpy(np.stack([windows[j][1] for j in batch_idx])).to(device)
            bb = torch.from_numpy(np.stack([windows[j][2] for j in batch_idx])).to(device)
            mb = torch.from_numpy(np.stack([windows[j][3] for j in batch_idx])).to(device)

            opt.zero_grad(set_to_none=True)
            lp2, lb2, lp1, lb1 = model(xb)
            denom = mb.sum().clamp_min(1.0)
            mb_pair = mb[:, 1:] * mb[:, :-1]
            denom_pair = mb_pair.sum().clamp_min(1.0)

            def stage_loss(lp, lb, tmse_w=0.0):
                # Keep calibrated BCE log-odds (no Focal logit compression before Viterbi)
                loss_p = (bce_poss(lp, yb) * mb).sum() / denom
                loss_b = (bce_bnd(lb, bb) * mb).sum() / denom
                if tmse_w > 0.0:
                    log_p1 = torch.nn.functional.logsigmoid(lp)
                    log_p0 = torch.nn.functional.logsigmoid(-lp)
                    d1 = (log_p1[:, 1:] - log_p1[:, :-1]).pow(2).clamp(max=16.0)
                    d0 = (log_p0[:, 1:] - log_p0[:, :-1]).pow(2).clamp(max=16.0)
                    loss_tmse = (0.5 * (d1 + d0) * mb_pair).sum() / denom_pair
                else:
                    loss_tmse = 0.0
                return loss_p + 0.35 * loss_b + tmse_w * loss_tmse

            loss = 0.4 * stage_loss(lp1, lb1, 0.0) + stage_loss(lp2, lb2, 0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def predict_seq_period(model, X, mean, std, win=SEQ_LEN, stride=SEQ_STRIDE_EVAL):
    """Slide window across one contiguous period and overlap-add with raised-cosine taper."""
    torch, _ = _import_torch()
    device = next(model.parameters()).device
    Xt = transform_features(X, mean, std)
    T = len(Xt)
    pA_sum = np.zeros(T, dtype=np.float64)
    pB_sum = np.zeros(T, dtype=np.float64)
    w_sum = np.zeros(T, dtype=np.float64)

    slices = _window_slices(T, win=win, stride=stride)
    # Use a raised-cosine (Hann + floor) weight per window so seams dissolve smoothly
    model.eval()
    with torch.no_grad():
        batch_x, batch_sl = [], []
        for s, e in slices:
            xw = Xt[s:e]
            batch_x.append(xw)
            batch_sl.append((s, e))
            if len(batch_x) == 32 or (s, e) == slices[-1]:
                # All slices in a long match period have length == win unless T < win
                t_len = batch_x[0].shape[0]
                taper = 0.1 + 0.9 * np.hanning(t_len) if t_len >= 16 else np.ones(t_len)
                xb = torch.from_numpy(np.stack(batch_x)).to(device)
                lp, lb = model(xb)
                pa = torch.sigmoid(lp).cpu().numpy()
                pb = torch.sigmoid(lb).cpu().numpy()
                for k, (ks, ke) in enumerate(batch_sl):
                    pA_sum[ks:ke] += pa[k] * taper
                    pB_sum[ks:ke] += pb[k] * taper
                    w_sum[ks:ke] += taper
                batch_x, batch_sl = [], []

    pA = (pA_sum / np.maximum(w_sum, 1e-6)).astype(np.float32)
    pB = (pB_sum / np.maximum(w_sum, 1e-6)).astype(np.float32)
    return pA, pB


def predict_seq_match(model, seq_list, mean, std):
    """Predict on a list of contiguous sequences matching features_by_period order."""
    pA_all, pB_all = [], []
    for s in seq_list:
        pa, pb = predict_seq_period(model, s["X"], mean, std)
        pA_all.append(pa)
        pB_all.append(pb)
    return pA_all, pB_all


# ----------------------------------------------------------------------------------------------
# 9. Validation & Full Training + Inference on BAR-ATH
# ----------------------------------------------------------------------------------------------


def eval_holdout_fold(g, tr_seqs, te_seqs, names, p_idx, seq_idx, device_choice="cpu"):
    from sklearn.metrics import roc_auc_score

    # 1. Diverse Tabular HGB ensemble (standard regularized + sharp leaf-127 touch tree) + Dead Ball
    Xtr, ytr = collect_tabular_live(tr_seqs, step=2)
    m_hgb1 = make_hgb(max_iter=450, min_samples_leaf=200, max_leaf_nodes=63, random_state=0).fit(Xtr, ytr)
    m_hgb2 = make_hgb(max_iter=450, min_samples_leaf=80, max_leaf_nodes=127, random_state=7).fit(Xtr, ytr)

    Xtr_d, ytr_d = collect_tabular_dead(tr_seqs, p_idx, step=3)
    m_dead = make_hgb(max_iter=350, min_samples_leaf=200).fit(Xtr_d, ytr_d)

    # 2. Heterogeneous PyTorch Sequence Ensemble: DeepResTCN (dilated conv) + PossessionSeqNet (BiGRU)
    mean, std = fit_scaler(tr_seqs)
    live_tr = _make_live_seq_view(tr_seqs, seq_idx)
    m_seq = train_seq_net(
        live_tr, mean[seq_idx], std[seq_idx],
        epochs=12, batch_size=96, lr=1.8e-3, seed=40 + g, device_choice=device_choice, arch="tcn",
    )
    m_conf = train_seq_net(
        live_tr, mean[seq_idx], std[seq_idx],
        epochs=8, batch_size=96, lr=2.0e-3, seed=240 + g, device_choice=device_choice, arch="bigru",
    )
    m_dseq = train_seq_net(
        _make_dead_seq_view(tr_seqs, p_idx),
        mean[p_idx], std[p_idx],
        epochs=8, batch_size=96, lr=2.0e-3, seed=140 + g, device_choice=device_choice, arch="bigru",
    )

    # Evaluate on hold-out copy 0 & 1 (one unswapped + one swapped real dropout realization)
    te_eval = [s for s in te_seqs if s["copy"] < 2]
    live_te = _make_live_seq_view(te_eval, seq_idx)
    p_seq_A1, p_seq_B1 = predict_seq_match(m_seq, live_te, mean[seq_idx], std[seq_idx])
    p_seq_A2, p_seq_B2 = predict_seq_match(m_conf, live_te, mean[seq_idx], std[seq_idx])
    p_seq_A = [0.5 * a1 + 0.5 * a2 for a1, a2 in zip(p_seq_A1, p_seq_A2)]
    p_seq_B = [np.maximum(b1, 0.5 * (b1 + b2)) for b1, b2 in zip(p_seq_B1, p_seq_B2)]

    y_live, vis_live, p_hgb_list, p_seq_list = [], [], [], []
    b_true, b_pred = [], []
    d_true, d_pred = [], []
    oof_periods = []
    bvis_col = names.index("bvis")
    dh1_col = names.index("dh1")
    da1_col = names.index("da1")
    bacc_col = names.index("bacc")

    p_dseq_all, p_dbnd_all = predict_seq_match(
        m_dseq, _make_dead_seq_view(te_eval, p_idx), mean[p_idx], std[p_idx]
    )
    for dseq_i, (s, psa, psb) in enumerate(zip(te_eval, p_seq_A, p_seq_B)):
        phgb = 0.5 * m_hgb1.predict_proba(s["X"])[:, 1] + 0.5 * m_hgb2.predict_proba(s["X"])[:, 1]
        pd_hgb = m_dead.predict_proba(s["X"][:, p_idx])[:, 1]
        pd_seq = p_dseq_all[dseq_i]
        pd_bnd = p_dbnd_all[dseq_i]
        pd_pr = 0.5 * pd_hgb + 0.5 * pd_seq
        lv = s["live"]
        dv = s["live"] | s["dead"]
        y_live.append(s["y"][lv] == 1)
        vis_live.append(s["X"][lv, bvis_col] > 0.5)
        p_hgb_list.append(phgb[lv])
        p_seq_list.append(psa[lv])
        b_true.append(s["bnd"][lv])
        b_pred.append(psb[lv])
        d_true.append(s["y"][dv] == 2)
        d_pred.append(pd_pr[dv])

        # Collect unswapped (copy == 0) period predictions for Metrica decoder calibration
        if s["copy"] == 0:
            p_hgb_h = (1.0 - phgb).astype(np.float32)
            p_seq_h = (1.0 - psa).astype(np.float32)
            p_home = 0.5 * (p_hgb_h + p_seq_h)
            oof_periods.append(dict(
                pH=p_home.astype(np.float32),
                pH_hgb=p_hgb_h,
                pH_seq=p_seq_h,
                pD=pd_pr.astype(np.float32),
                pD_hgb=pd_hgb.astype(np.float32),
                pD_seq=pd_seq.astype(np.float32),
                pD_bnd=pd_bnd.astype(np.float32),
                pBound=psb.astype(np.float32),
                y=s["y_ch"],
                raw_vis=s["X"][:, bvis_col] > 0.5,
                dh1=s["X"][:, dh1_col],
                da1=s["X"][:, da1_col],
                bacc=s["X"][:, bacc_col],
            ))

    yt = np.concatenate(y_live)
    vis = np.concatenate(vis_live)
    ph = np.concatenate(p_hgb_list)
    ps = np.concatenate(p_seq_list)
    pe = 0.5 * ph + 0.5 * ps
    bt = np.concatenate(b_true)
    bp = np.concatenate(b_pred)
    dt = np.concatenate(d_true)
    dp = np.concatenate(d_pred)

    def acc_stats(prob):
        return (
            float(np.mean((prob > 0.5) == yt)),
            float(np.mean(((prob > 0.5) == yt)[vis])),
            float(np.mean(((prob > 0.5) == yt)[~vis])),
        )

    ah = acc_stats(ph)
    asq = acc_stats(ps)
    ae = acc_stats(pe)
    d_acc = float(np.mean((dp > 0.5) == dt))
    d_auc = float(roc_auc_score(dt, dp))
    b_auc = float(roc_auc_score(bt > 0.5, bp))
    b_sw = float(bp[bt > 0.5].mean())
    b_st = float(bp[bt < 0.5].mean())
    print(
        "  hold-out game {}: HGB {:.3f} (vis {:.3f}/miss {:.3f}) | "
        "SeqNet {:.3f} (vis {:.3f}/miss {:.3f}) | "
        "ENS {:.3f} (vis {:.3f}/miss {:.3f})".format(
            g, ah[0], ah[1], ah[2], asq[0], asq[1], asq[2], ae[0], ae[1], ae[2]
        )
    )
    print(
        "    -> Dead(player-only): acc {:.3f}, AUC {:.3f} | "
        "Boundary: AUC {:.3f}, P(switch)={:.3f} vs P(steady)={:.3f}".format(
            d_acc, d_auc, b_auc, b_sw, b_st
        )
    )
    return oof_periods


def calibrate_dead_stage1(oof_periods, gt_all):
    """Calibrate Stage-1 Dead-vs-Live blend & Viterbi parameters strictly on Metrica OOF."""
    from sklearn.metrics import roc_auc_score
    gt_live_bin = (gt_all == pm.D).astype(np.int8)
    scored = gt_all != pm.X
    gt_b = gt_live_bin[scored]

    has_split = "pD_hgb" in oof_periods[0] and "pD_seq" in oof_periods[0]
    has_bnd = "pD_bnd" in oof_periods[0]
    if has_split:
        p_hgb_all = np.concatenate([p["pD_hgb"] for p in oof_periods])[scored]
        p_seq_all = np.concatenate([p["pD_seq"] for p in oof_periods])[scored]
        print("  Metrica OOF Dead AUC: HGB={:.4f} | DeadSeqNet={:.4f} | 50/50={:.4f}".format(
            roc_auc_score(gt_b, p_hgb_all),
            roc_auc_score(gt_b, p_seq_all),
            roc_auc_score(gt_b, 0.5 * p_hgb_all + 0.5 * p_seq_all),
        ))
        w_grid = (0.50, 0.65, 0.75, 0.85, 1.00)
    else:
        w_grid = (0.50,)
    alpha_grid = (0.0, 1.0, 2.0, 3.0) if has_bnd else (0.0,)

    best_dead = None
    best_extra = {"dead_seq_weight": 0.5, "dead_boundary_alpha": 0.0}
    best_dead_key = (-1.0, -99.0, -99.0, -1.0)
    best_dead_score = (-1.0, -1.0)
    best_masks = None

    for w in w_grid:
        if has_split:
            p_w = w * p_seq_all + (1.0 - w) * p_hgb_all
            brier_w = float(np.mean((p_w - gt_b) ** 2))
            d_logits = [
                logit_from_prob(w * p["pD_seq"] + (1.0 - w) * p["pD_hgb"])
                for p in oof_periods
            ]
        else:
            brier_w = 0.0
            d_logits = [logit_from_prob(p["pD"]) for p in oof_periods]
        d_bnds = [p["pD_bnd"] for p in oof_periods] if has_bnd else [None] * len(oof_periods)

        for d_alpha in alpha_grid:
            for bias in (0.0,):
                for sw in (15.0, 20.0, 25.0, 35.0):
                    raw_dead = [
                        viterbi_binary(dl + bias, sw, boundary_prob=db, boundary_alpha=d_alpha)
                        for dl, db in zip(d_logits, d_bnds)
                    ]
                    for min_d in (25, 50):
                        segs = [
                            absorb_short_runs(rd, min_d) if min_d > 1 else rd
                            for rd in raw_dead
                        ]
                        pred_d = np.concatenate(segs)
                        pr_b = pred_d[scored]
                        tp = int(np.sum((gt_b == 1) & (pr_b == 1)))
                        fp = int(np.sum((gt_b == 0) & (pr_b == 1)))
                        fn = int(np.sum((gt_b == 1) & (pr_b == 0)))
                        f1_dead = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
                        tn = int(np.sum((gt_b == 0) & (pr_b == 0)))
                        f1_live = (2.0 * tn / (2.0 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
                        ad_f1 = 0.5 * (f1_dead + f1_live)
                        # Prefer higher ad_f1, stronger boundary sensitivity (d_alpha), higher sw, lower False Dead (fp)
                        key = (round(ad_f1, 3), float(d_alpha), float(sw), -fp)
                        if key > best_dead_key:
                            best_dead_key = key
                            best_dead_score = (ad_f1, f1_dead)
                            best_dead = (bias, sw, min_d)
                            best_extra = {
                                "dead_seq_weight": float(w),
                                "dead_boundary_alpha": float(d_alpha),
                            }
                            best_masks = [s.astype(bool) for s in segs]

    print(
        "  Stage-1 blend w_seq={:.2f}, d_alpha={:.1f}".format(
            best_extra["dead_seq_weight"], best_extra["dead_boundary_alpha"]
        )
    )
    calibrate_dead_stage1.last_extra = best_extra
    return best_dead, best_dead_score, best_masks


calibrate_dead_stage1.last_extra = {"dead_seq_weight": 0.5, "dead_boundary_alpha": 0.0}


def blend_live_logits(p_seq, p_hgb, vis_mask, w_vis=0.5, w_miss=0.5, mode="prob"):
    """Blend SeqNet and HGB P(Home|live) conditioned on ball visibility, returning logits."""
    eps = 1e-5
    ps = np.clip(np.asarray(p_seq, dtype=np.float64), eps, 1.0 - eps)
    ph = np.clip(np.asarray(p_hgb, dtype=np.float64), eps, 1.0 - eps)
    w = np.where(np.asarray(vis_mask, dtype=bool), float(w_vis), float(w_miss))
    if mode == "logit":
        ls = np.log(ps / (1.0 - ps))
        lh = np.log(ph / (1.0 - ph))
        return (w * ls + (1.0 - w) * lh).astype(np.float64)
    pb = np.clip(w * ps + (1.0 - w) * ph, eps, 1.0 - eps)
    return np.log(pb / (1.0 - pb)).astype(np.float64)


def _eval_stage2_oof(oof_periods, h_logits, dead_masks, gt_all, gt_runs, sw, b_alpha, min_sp):
    """Evaluate a Stage-2 Viterbi + post-processing config on Metrica OOF."""
    pred_parts = []
    for p, hl, dm in zip(oof_periods, h_logits, dead_masks):
        live_mask = ~dm
        padded = np.concatenate(([False], live_mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        codes = np.where(dm, pm.D, pm.H).astype(np.int8)
        for s_idx, e_idx in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
            st = viterbi_binary(
                hl[s_idx:e_idx],
                sw,
                boundary_prob=p["pBound"][s_idx:e_idx],
                boundary_alpha=b_alpha,
            )
            cur = absorb_short_runs(st, min_sp) if min_sp > 1 else st.copy()
            codes[s_idx:e_idx] = np.where(cur == 1, pm.H, pm.A)
        pred_parts.append(codes)
    pred_all = np.concatenate(pred_parts)
    cm3 = pm.scored_confusion(pm.confusion(gt_all, pred_all))
    sf1 = pm.spell_scores(gt_all, pred_all, gt_runs)["mean_f1"]
    mf1 = pm.macro_f1(cm3)
    return sf1, mf1


def calibrate_decoder_on_metrica(oof_periods):
    """Select visibility gating, Viterbi & post-processing hyperparameters strictly on Metrica OOF."""
    print("Calibrating decoder hyperparameters on Metrica Sample Games 1-3 OOF...")
    gt_all = np.concatenate([
        np.where(p["y"] >= 0, p["y"], pm.X).astype(np.int8) for p in oof_periods
    ])
    gt_runs = pm.valid_runs(gt_all)

    # Stage 1: Dead-ball Viterbi calibration (maximising alive_dead_f1, tie-break dead F1)
    best_dead, best_dead_score, dead_masks = calibrate_dead_stage1(oof_periods, gt_all)

    dead_bias, dead_sw, min_dead = best_dead
    print(
        "  Best Dead decoder (Metrica OOF): bias={:+.2f}, switch={:.1f}, min_dead={} -> "
        "alive_dead_f1={:.4f}, dead_f1={:.4f}".format(
            dead_bias, dead_sw, min_dead, best_dead_score[0], best_dead_score[1]
        )
    )

    # Stage 2A: Visibility-conditioned SeqNet vs HGB gating calibration
    # Bounded in [0.40, 0.60] probability space to preserve GBDT + Neural ensemble diversity under domain shift
    has_ha_split = "pH_hgb" in oof_periods[0] and "pH_seq" in oof_periods[0]
    if has_ha_split:
        blend_cands = [(0.50, 0.50, "prob")]
        for wv in (0.50, 0.55, 0.60):
            for wm in (0.40, 0.45, 0.50):
                blend_cands.append((wv, wm, "prob"))

        scored_blends = []
        for wv, wm, mode in blend_cands:
            hls = [
                blend_live_logits(p["pH_seq"], p["pH_hgb"], p["raw_vis"], wv, wm, mode)
                for p in oof_periods
            ]
            sf1, mf1 = _eval_stage2_oof(oof_periods, hls, dead_masks, gt_all, gt_runs, 24.0, 0.7, 38)
            scored_blends.append(((sf1, mf1), (wv, wm, mode), hls))
        scored_blends.sort(reverse=True, key=lambda t: t[0])
        top_blends = scored_blends[:4]
        print(
            "  Top visibility blend (mid-decoder): w_vis={:.2f}, w_miss={:.2f}, mode={} -> spellF1={:.4f}, macroF1={:.4f}".format(
                top_blends[0][1][0], top_blends[0][1][1], top_blends[0][1][2],
                top_blends[0][0][0], top_blends[0][0][1],
            )
        )
    else:
        hls = [logit_from_prob(p["pH"]) for p in oof_periods]
        top_blends = [((0.0, 0.0), (0.50, 0.50, "prob"), hls)]

    # Stage 2B: Joint Stage-2 decoder sweep across top visibility-gated blends
    w_d = float(calibrate_dead_stage1.last_extra.get("dead_seq_weight", 0.75))
    best_fixed = None
    best_fixed_blend = (0.50, 0.50, "prob")
    best_fixed_score = (-1.0, -1.0)
    best_bg = None
    best_bg_blend = (0.50, 0.50, "prob")
    best_bg_score = (-1.0, -1.0)
    for _, blend_cfg, h_logits in top_blends:
        for sw in (10.0, 12.0, 14.0, 18.0, 24.0, 30.0):
            for b_alpha in (0.0, 0.4, 0.7, 1.0):
                per_period_raw = []
                for p, hl, dm in zip(oof_periods, h_logits, dead_masks):
                    pD_p = (w_d * p["pD_seq"] + (1.0 - w_d) * p["pD_hgb"]) if "pD_seq" in p else p["pD"]
                    live_mask = ~dm
                    padded = np.concatenate(([False], live_mask, [False]))
                    diff = np.diff(padded.astype(np.int8))
                    segs = []
                    for s_idx, e_idx in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
                        st = viterbi_binary(
                            hl[s_idx:e_idx],
                            sw,
                            boundary_prob=p["pBound"][s_idx:e_idx],
                            boundary_alpha=b_alpha,
                        )
                        segs.append((int(s_idx), int(e_idx), st))
                    per_period_raw.append((p, hl, dm, pD_p, segs))

                for min_sp in (38, 50):
                    bg_cands = [(None, 0.50, None, 0, 1.0)]
                    for s_min in (25, 30):
                        for l_thr in (2.2, 2.5, 2.8):
                            bg_cands.append((s_min, 0.50, l_thr, 0, 1.0))
                            for suf_len, suf_thr in ((150, 0.20), (175, 0.20), (175, 0.22)):
                                bg_cands.append((s_min, 0.50, l_thr, suf_len, suf_thr))
                    for short_min_sp, bnd_thr, l_thr, suf_len, suf_thr in bg_cands:
                        pred_parts = []
                        for p, hl_p, dm, pD_p, segs in per_period_raw:
                            codes = np.where(dm, pm.D, pm.H).astype(np.int8)
                            n_tot = len(codes)
                            for s_idx, e_idx, st in segs:
                                seg_ed = hl_p[s_idx:e_idx]
                                seg_pD = pD_p[s_idx:e_idx]
                                if min_sp > 1:
                                    cur = absorb_short_runs(
                                        st, min_sp,
                                        boundary_prob=p["pBound"][s_idx:e_idx],
                                        short_min_len=short_min_sp,
                                        boundary_thresh=bnd_thr,
                                        emission_diff=seg_ed,
                                        logit_thresh=l_thr,
                                        dead_prob=seg_pD,
                                        suffix_dead_thr=suf_thr,
                                    )
                                else:
                                    cur = st.copy()
                                seg_codes = np.where(cur == 1, pm.H, pm.A).astype(np.int8)
                                if suf_len > 0 and e_idx < n_tot and len(cur) > 0:
                                    breaks = np.flatnonzero(cur[1:] != cur[:-1]) + 1
                                    bounds = np.concatenate(([0], breaks, [len(cur)]))
                                    if len(bounds) - 1 >= 2:
                                        rs, re = int(bounds[-2]), int(bounds[-1])
                                        sign = 1.0 if cur[rs] == 1 else -1.0
                                        mean_l = float(np.mean(sign * seg_ed[rs:re]))
                                        if (re - rs) <= suf_len and float(np.mean(seg_pD[rs:re])) >= suf_thr and (l_thr is None or mean_l < l_thr):
                                            seg_codes[rs:re] = pm.D
                                codes[s_idx:e_idx] = seg_codes
                            pred_parts.append(codes)
                        pred_all = np.concatenate(pred_parts)
                        cm3 = pm.scored_confusion(pm.confusion(gt_all, pred_all))
                        sp_res = pm.spell_scores(gt_all, pred_all, gt_runs)
                        sc = (sp_res["mean_f1"], pm.macro_f1(cm3))
                        if short_min_sp is None and suf_len == 0:
                            if sc > best_fixed_score:
                                best_fixed_score = sc
                                best_fixed = (sw, b_alpha, min_sp, None, 0.50, None, 0, 1.0)
                                best_fixed_blend = blend_cfg
                        else:
                            if sc > best_bg_score:
                                best_bg_score = sc
                                best_bg = (sw, b_alpha, min_sp, short_min_sp, bnd_thr, l_thr, suf_len, suf_thr)
                                best_bg_blend = blend_cfg

    # Accept confidence/boundary-gated min_spell only if Metrica OOF spellF1 improves >= 0.002 without macroF1 drop
    if best_bg is not None and best_bg_score[0] >= best_fixed_score[0] + 0.002 and best_bg_score[1] >= best_fixed_score[1]:
        best_live, best_blend, best_live_score = best_bg, best_bg_blend, best_bg_score
    else:
        best_live, best_blend, best_live_score = best_fixed, best_fixed_blend, best_fixed_score

    live_sw, b_alpha, min_sp, short_min_sp, short_bnd_thr, short_logit_thr, suf_len, suf_thr = best_live
    w_vis, w_miss, blend_mode = best_blend
    print(
        "  Best Live decoder (Metrica OOF): w_vis={:.2f}, w_miss={:.2f} ({}), switch={:.1f}, "
        "boundary_alpha={:.2f}, min_spell={}, short_min={} (logit_thr={}), suf_dead=({},{:.2f}) -> Metrica OOF spellF1={:.4f}, macroF1={:.4f}".format(
            w_vis, w_miss, blend_mode, live_sw, b_alpha, min_sp, short_min_sp, short_logit_thr,
            suf_len, suf_thr, best_live_score[0], best_live_score[1]
        )
    )
    return {
        "calibrated_on": "Metrica Sample Games 1-3 (Leave-One-Game-Out CV)",
        "dead_logit_bias": float(dead_bias),
        "dead_switch_penalty": float(dead_sw),
        "min_dead_frames": int(min_dead),
        "live_seq_weight_vis": float(w_vis),
        "live_seq_weight_miss": float(w_miss),
        "live_blend_mode": str(blend_mode),
        "live_switch_penalty": float(live_sw),
        "boundary_alpha": float(b_alpha),
        "min_spell_frames": int(min_sp),
        "short_min_spell_frames": int(short_min_sp) if short_min_sp is not None else None,
        "short_boundary_thresh": float(short_bnd_thr),
        "short_logit_thresh": float(short_logit_thr) if short_logit_thr is not None else None,
        "suffix_dead_max_frames": int(suf_len),
        "suffix_dead_pD_thresh": float(suf_thr),
        "metrica_oof_alive_dead_f1": float(best_dead_score[0]),
        "metrica_oof_spell_f1_mean": float(best_live_score[0]),
        "metrica_oof_macro_f1": float(best_live_score[1]),
    }


def _four_fold_bar_ath_features(bh, ba, bball, bper):
    """Compute features over the 4 attack-preserving football symmetries (D4 subgroup)."""
    h, a, b = normalise_direction(bh, ba, bball, bper)
    symmetries = []
    for swap_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
        if not swap_x:
            hc, ac, bc = h.copy(), a.copy(), b.copy()
        else:
            hc, ac, bc = a.copy(), h.copy(), b.copy()
            hc[..., 0] = L - hc[..., 0]
            ac[..., 0] = L - ac[..., 0]
            bc[:, 0] = L - bc[:, 0]
        if flip_y:
            hc[..., 1] = W - hc[..., 1]
            ac[..., 1] = W - ac[..., 1]
            bc[:, 1] = W - bc[:, 1]
        X_sym, names_sym = features_by_period(hc, ac, bc, bper)
        symmetries.append((swap_x, X_sym, names_sym))
    return symmetries


def predict_bar_ath(
    m_hgb, m_dead, m_seq, names, p_idx, seq_idx, mean, std, bh, ba, bball, bper, blive,
    m_hgb2=None, m_conf=None,
):
    """Run deterministic 4-fold symmetry TTA inference on BAR-ATH and save cache/metrica_ml_*.npy."""
    print("Predicting on BAR-ATH with 4-fold symmetry Test-Time Augmentation...")
    syms = _four_fold_bar_ath_features(bh, ba, bball, bper)
    assert syms[0][2] == names

    T = len(bper)
    pH_hgb_sum = np.zeros(T, dtype=np.float64)
    pH_seq_sum = np.zeros(T, dtype=np.float64)
    pD_hgb_sum = np.zeros(T, dtype=np.float64)
    pB_sum = np.zeros(T, dtype=np.float64)
    s_mean, s_std = mean[seq_idx], std[seq_idx]

    for swap_x, X_sym, _ in syms:
        p1_hgb = m_hgb.predict_proba(X_sym)[:, 1]
        if m_hgb2 is not None:
            p1_hgb = 0.5 * p1_hgb + 0.5 * m_hgb2.predict_proba(X_sym)[:, 1]
        pH_hgb_sum += p1_hgb if swap_x else (1.0 - p1_hgb)
        pD_hgb_sum += m_dead.predict_proba(X_sym[:, p_idx])[:, 1]
        for p in np.unique(bper):
            m = np.flatnonzero(bper == p)
            pa, pb = predict_seq_period(m_seq, X_sym[m][:, seq_idx], s_mean, s_std)
            if m_conf is not None:
                pa_c, pb_c = predict_seq_period(m_conf, X_sym[m][:, seq_idx], s_mean, s_std)
                pa = 0.5 * pa + 0.5 * pa_c
                pb = np.maximum(pb, 0.5 * (pb + pb_c))
            pH_seq_sum[m] += pa if swap_x else (1.0 - pa)
            pB_sum[m] += pb

    pH_hgb = (pH_hgb_sum / len(syms)).astype(np.float32)
    pH_seq = (pH_seq_sum / len(syms)).astype(np.float32)
    pD = (pD_hgb_sum / len(syms)).astype(np.float32)
    pB = (pB_sum / len(syms)).astype(np.float32)
    pH = (0.5 * pH_hgb + 0.5 * pH_seq).astype(np.float32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pH_hgb.npy"), pH_hgb)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pH_seq.npy"), pH_seq)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pH.npy"), pH)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pD.npy"), pD)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pBound.npy"), pB)

    print(
        "  BAR-ATH live frames: mean P(H)={:.3f} (HGB {:.3f} / SeqNet {:.3f}), "
        "mean P(D | live)={:.3f}, mean P(D | ~live)={:.3f}, P(Bound) p95={:.3f}".format(
            pH[blive].mean(), pH_hgb[blive].mean(), pH_seq[blive].mean(),
            pD[blive].mean(), pD[~blive].mean(), np.percentile(pB[blive], 95),
        )
    )
    return pH, pD, pB, syms


def load_saved_models(device_choice="cpu"):
    """Load pre-trained Metrica models from models/ for deterministic inference."""
    import joblib
    torch, _ = _import_torch()
    dev = get_device(torch, device_choice)
    bundle = joblib.load(os.path.join(MODEL_DIR, "metrica_models.joblib"))
    names = bundle["names"]
    p_idx = bundle["p_idx"]
    seq_idx = bundle.get("seq_idx", list(range(len(names))))
    mean = bundle["mean"]
    std = bundle["std"]
    seq_arch = bundle.get("seq_arch", "bigru")
    m_seq = build_seq_net(2 * len(seq_idx), arch=seq_arch).to(dev)
    state = torch.load(os.path.join(MODEL_DIR, "metrica_seqnet.pt"), map_location=dev)
    m_seq.load_state_dict(state)
    m_seq.eval()
    conf_path = os.path.join(MODEL_DIR, "metrica_confnet.pt")
    m_conf = None
    if os.path.isfile(conf_path):
        conf_arch = bundle.get("conf_arch", "bigru")
        m_conf = build_seq_net(2 * len(seq_idx), arch=conf_arch).to(dev)
        m_conf.load_state_dict(torch.load(conf_path, map_location=dev))
        m_conf.eval()
    dseq_path = os.path.join(MODEL_DIR, "metrica_deadseqnet.pt")
    m_dseq = None
    if os.path.isfile(dseq_path):
        dseq_arch = bundle.get("deadseq_arch", "bigru")
        m_dseq = build_seq_net(2 * len(p_idx), arch=dseq_arch).to(dev)
        m_dseq.load_state_dict(torch.load(dseq_path, map_location=dev))
        m_dseq.eval()
    return (
        bundle["m_hgb"], bundle["m_dead"], m_seq, m_dseq,
        names, p_idx, seq_idx, mean, std, bundle.get("m_hgb2"), m_conf,
    )


def predict_dead_seq_bar_ath(m_dseq, p_idx, mean, std, bh, ba, bball, bper, dead_seq_weight=0.5, syms=None):
    """Symmetric 4-fold inference for temporal DeadSeqNet on BAR-ATH player-only features."""
    if m_dseq is None:
        return
    if syms is None:
        syms = _four_fold_bar_ath_features(bh, ba, bball, bper)

    T = len(bper)
    pD_sum = np.zeros(T, dtype=np.float64)
    pB_sum = np.zeros(T, dtype=np.float64)
    d_mean, d_std = mean[p_idx], std[p_idx]
    for _, X_sym, _ in syms:
        for p in np.unique(bper):
            m = np.flatnonzero(bper == p)
            pr_d, pr_b = predict_seq_period(m_dseq, X_sym[m][:, p_idx], d_mean, d_std)
            pD_sum[m] += pr_d
            pB_sum[m] += pr_b
    pD_seq = (pD_sum / len(syms)).astype(np.float32)
    pD_bnd = (pB_sum / len(syms)).astype(np.float32)

    pD_hgb = np.load(os.path.join(CACHE_DIR, "metrica_ml_pD.npy")).astype(np.float32)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pD_hgb.npy"), pD_hgb)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pD_seq.npy"), pD_seq)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pD_bnd.npy"), pD_bnd)
    w = float(dead_seq_weight)
    pD_blend = (w * pD_seq + (1.0 - w) * pD_hgb).astype(np.float32)
    np.save(os.path.join(CACHE_DIR, "metrica_ml_pD.npy"), pD_blend)


def save_oof_cache(oof_periods, path):
    """Save Metrica OOF period predictions to disk so calibration can re-run in seconds."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"n_periods": np.array([len(oof_periods)], dtype=np.int32)}
    keys = (
        "pH", "pH_hgb", "pH_seq", "pD", "pD_hgb", "pD_seq", "pD_bnd", "pBound",
        "y", "raw_vis", "dh1", "da1", "bacc",
    )
    for i, p in enumerate(oof_periods):
        for k in keys:
            if k in p:
                arrays["p{}_{}".format(i, k)] = p[k]
    np.savez_compressed(path, **arrays)


def load_oof_cache(path):
    """Load cached Metrica OOF period predictions."""
    data = np.load(path)
    n = int(data["n_periods"][0])
    keys = (
        "pH", "pH_hgb", "pH_seq", "pD", "pD_hgb", "pD_seq", "pD_bnd", "pBound",
        "y", "raw_vis", "dh1", "da1", "bacc",
    )
    out = []
    for i in range(n):
        d = {}
        for k in keys:
            ak = "p{}_{}".format(i, k)
            if ak in data:
                d[k] = data[ak]
        out.append(d)
    return out


def _read_dead_seq_weight():
    dp_path = os.path.join(MODEL_DIR, "decoder_params.json")
    if os.path.isfile(dp_path):
        with open(dp_path) as fh:
            return float(json.load(fh).get("dead_seq_weight", 0.5))
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="Skip LOGO CV & decoder calibration")
    ap.add_argument("--predict-only", action="store_true", help="Deterministic inference from saved models/")
    ap.add_argument("--calibrate-only", action="store_true", help="Run LOGO CV to calibrate decoder_params.json only")
    ap.add_argument("--recompute-oof", action="store_true", help="Ignore cached metrica_oof.npz and rerun CV")
    ap.add_argument("--device", default="cpu", choices=("cpu", "auto", "mps", "cuda"))
    args = ap.parse_args()
    t0 = time.time()
    set_deterministic(2027)

    if args.predict_only:
        print("Running deterministic inference from saved models/ ...")
        bh, ba, bball, bper, blive, _ = load_bar_ath()
        m_hgb, m_dead, m_seq, m_dseq, names, p_idx, seq_idx, mean, std, m_hgb2, m_conf = load_saved_models(args.device)
        _, _, _, syms = predict_bar_ath(
            m_hgb, m_dead, m_seq, names, p_idx, seq_idx, mean, std, bh, ba, bball, bper, blive,
            m_hgb2=m_hgb2, m_conf=m_conf,
        )
        predict_dead_seq_bar_ath(
            m_dseq, p_idx, mean, std, bh, ba, bball, bper, dead_seq_weight=_read_dead_seq_weight(), syms=syms
        )
        print("Saved metrica_ml_pH.npy, metrica_ml_pD.npy, metrica_ml_pBound.npy. Done in {:.0f}s".format(time.time() - t0))
        return

    oof_cache_path = os.environ.get("POSSESSION_OOF_PATH") or os.path.join(CACHE_DIR, "metrica_oof.npz")
    if args.calibrate_only and os.path.isfile(oof_cache_path) and not args.recompute_oof:
        print("Loading cached Metrica OOF predictions from {} ...".format(oof_cache_path))
        oof_periods = load_oof_cache(oof_cache_path)
        decoder_params = calibrate_decoder_on_metrica(oof_periods)
        decoder_params.update(calibrate_dead_stage1.last_extra)
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "decoder_params.json"), "w") as fh:
            json.dump(decoder_params, fh, indent=2)
        print("Saved models/decoder_params.json in {:.0f}s".format(time.time() - t0))
        return

    print("BAR-ATH: measuring ball dropout run lengths inside detected live play (no labels)...")
    bh, ba, bball, bper, blive, bvis = load_bar_ath()
    vis_runs, miss_runs = bar_ath_run_lengths(blive, bvis)
    print("  live ball visible fraction {:.1%}; visible runs median {} / missing runs median {} frames".format(
        bvis[blive].mean(), int(np.median(vis_runs)), int(np.median(miss_runs))))

    games = {}
    for g in (1, 2, 3):
        print("Loading Metrica Sample Game {} ...".format(g))
        games[g] = load_metrica_game(g)
        lab = event_labels(games[g]["events"], games[g]["period"], len(games[g]["ball"]))
        print("  frames {:,}  H {:.1%}  A {:.1%}  D {:.1%}  ignored {:.1%}".format(
            len(lab), np.mean(lab == 0), np.mean(lab == 1), np.mean(lab == 2), np.mean(lab == -1)))

    game_seqs = {}
    names = None
    for g in games:
        seqs, names = build_game_copies(games[g], vis_runs, miss_runs, seed=100 + g, n_copies=8)
        game_seqs[g] = seqs
        live_rows = sum(int(s["live"].sum()) for s in seqs)
        dead_rows = sum(int(s["dead"].sum()) for s in seqs)
        print("  game {} -> {} period sequences | live rows {:,} | dead rows {:,}".format(
            g, len(seqs), live_rows, dead_rows))

    p_idx = player_only_indices(names)
    seq_idx = seq_live_indices(names)
    print("Features: {} total, {} pitch-invariant (SeqNet), {} player-only (Dead Ball)".format(
        len(names), len(seq_idx), len(p_idx)
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    if not args.no_cv:
        print("Leave-one-game-out validation across Metrica Sample Games 1-3:")
        oof_periods = []
        for g in games:
            tr_seqs = [s for k in games if k != g for s in game_seqs[k]]
            oof_periods.extend(
                eval_holdout_fold(g, tr_seqs, game_seqs[g], names, p_idx, seq_idx, device_choice=args.device)
            )
        save_oof_cache(oof_periods, oof_cache_path)
        decoder_params = calibrate_decoder_on_metrica(oof_periods)
        decoder_params.update(calibrate_dead_stage1.last_extra)
        with open(os.path.join(MODEL_DIR, "decoder_params.json"), "w") as fh:
            json.dump(decoder_params, fh, indent=2)
        print("Saved models/decoder_params.json")

    if args.calibrate_only:
        print("Calibration complete in {:.0f}s".format(time.time() - t0))
        return

    print("Training final models on all 3 Metrica games (device={})...".format(args.device))
    all_seqs = [s for g in games for s in game_seqs[g]]
    Xtr, ytr = collect_tabular_live(all_seqs, step=2)
    m_hgb = make_hgb(max_iter=500, min_samples_leaf=200, max_leaf_nodes=63, random_state=0).fit(Xtr, ytr)
    m_hgb2 = make_hgb(max_iter=500, min_samples_leaf=80, max_leaf_nodes=127, random_state=7).fit(Xtr, ytr)

    Xtr_d, ytr_d = collect_tabular_dead(all_seqs, p_idx, step=3)
    m_dead = make_hgb(max_iter=400, min_samples_leaf=200).fit(Xtr_d, ytr_d)

    mean, std = fit_scaler(all_seqs)
    live_all = _make_live_seq_view(all_seqs, seq_idx)
    m_seq = train_seq_net(
        live_all, mean[seq_idx], std[seq_idx],
        epochs=14, batch_size=96, lr=1.8e-3, seed=2027, device_choice=args.device, arch="tcn",
    )
    m_conf = train_seq_net(
        live_all, mean[seq_idx], std[seq_idx],
        epochs=10, batch_size=96, lr=2.0e-3, seed=2227, device_choice=args.device, arch="bigru",
    )
    m_dseq = train_seq_net(
        _make_dead_seq_view(all_seqs, p_idx),
        mean[p_idx], std[p_idx],
        epochs=10, batch_size=96, lr=2.0e-3, seed=2127, device_choice=args.device, arch="bigru",
    )

    _, _, _, syms = predict_bar_ath(
        m_hgb, m_dead, m_seq, names, p_idx, seq_idx, mean, std, bh, ba, bball, bper, blive,
        m_hgb2=m_hgb2, m_conf=m_conf,
    )
    predict_dead_seq_bar_ath(
        m_dseq, p_idx, mean, std, bh, ba, bball, bper, dead_seq_weight=_read_dead_seq_weight(), syms=syms
    )

    try:
        import joblib
        torch, _ = _import_torch()
        joblib.dump(
            dict(
                m_hgb=m_hgb, m_hgb2=m_hgb2, m_dead=m_dead, names=names, p_idx=p_idx, seq_idx=seq_idx,
                mean=mean, std=std, seq_arch="tcn", conf_arch="bigru", deadseq_arch="bigru",
            ),
            os.path.join(MODEL_DIR, "metrica_models.joblib"),
        )
        torch.save(m_seq.state_dict(), os.path.join(MODEL_DIR, "metrica_seqnet.pt"))
        torch.save(m_conf.state_dict(), os.path.join(MODEL_DIR, "metrica_confnet.pt"))
        torch.save(m_dseq.state_dict(), os.path.join(MODEL_DIR, "metrica_deadseqnet.pt"))
    except Exception as exc:  # pragma: no cover
        print("  (model save skipped: {})".format(exc))

    print("Saved metrica_ml_pH.npy, metrica_ml_pD.npy, metrica_ml_pBound.npy. Done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
