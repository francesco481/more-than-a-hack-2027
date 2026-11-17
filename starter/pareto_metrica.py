"""Pareto Multi-Objective Optimization strictly across Metrica Sample Games 1-3.

This script finds the non-dominated Pareto frontier across the 3 independent Metrica matches:
  Objective 1: spellF1(Game 1)
  Objective 2: spellF1(Game 2)
  Objective 3: spellF1(Game 3)
  Tie-breakers: Pooled macroF1 and worst-case game robustness min(J1, J2, J3).

BAR-ATH ground truth (public 5-min or full match) is NEVER loaded or read.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from starter import algorithmic_baseline as ab  # noqa: E402
from starter import metrica_possession_model as mpm  # noqa: E402
from starter.possession_decoder import absorb_short_runs, logit_from_prob, viterbi_binary  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
MODEL_DIR = os.path.join(ROOT, "models")

N_GAMES = 3
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def decode_period(p, P):
    """One Metrica period -> label codes, mirroring the BAR-ATH two-stage decoder."""
    T = len(p["y"])
    w_d = float(P["dead_seq_weight"])
    pD = (w_d * p["pD_seq"] + (1.0 - w_d) * p["pD_hgb"]).astype(np.float32)
    st = viterbi_binary(
        logit_from_prob(pD) + float(P["dead_logit_bias"]), float(P["dead_switch_penalty"]),
        boundary_prob=p["pD_bnd"], boundary_alpha=float(P["dead_boundary_alpha"]))
    if int(P["min_dead_frames"]) > 1:
        st = absorb_short_runs(st, int(P["min_dead_frames"]))
    dead = st.astype(bool)

    live_logit = mpm.blend_live_logits(
        p["pH_seq"], p["pH_hgb"], p["raw_vis"], w_vis=float(P["live_seq_weight_vis"]),
        w_miss=float(P["live_seq_weight_miss"]), mode=str(P["live_blend_mode"])).astype(np.float32)

    ab.half_video_blocks = lambda game: [("period", 0, T - 1)]
    labels = ab.decode_live_possessions(
        live_logit, dead, np.asarray(p["raw_vis"], bool), p["dh1"], p["da1"], p["pBound"], P,
        game=None, ml_pD=pD, pH_hgb=p["pH_hgb"], local_logit=live_logit, sp_restart_windows=None)
    out = np.full(T, pm.X, np.int8)
    for k, c in CODE.items():
        out[labels == k] = c
    return out


class Bench:
    def __init__(self, path):
        self.periods = mpm.load_oof_cache(path)
        assert len(self.periods) == 2 * N_GAMES, "expected two periods per Metrica game"
        assert "pH_seq" in self.periods[0], "cache lacks pH_hgb / pH_seq; rerun --calibrate-only --recompute-oof"
        self.gt = [np.where(p["y"] >= 0, p["y"], pm.X).astype(np.int8) for p in self.periods]

    def score(self, P, games):
        sep = np.array([pm.X], np.int8)
        g, q = [], []
        for i, (p, gt) in enumerate(zip(self.periods, self.gt)):
            if i // 2 in games:
                g += [gt, sep]
                q += [decode_period(p, P), sep]
        g, q = np.concatenate(g), np.concatenate(q)
        sf1 = pm.spell_scores(g, q, pm.valid_runs(g))["mean_f1"]
        mf1 = pm.macro_f1(pm.scored_confusion(pm.confusion(g, q)))
        return float(sf1), float(mf1)


def is_pareto_efficient(costs):
    """Find Pareto-efficient points (maximizing all objectives).

    costs: (N, d) array of objectives to maximize.
    returns boolean array of length N.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower value; mark as dominated if strictly beaten in all objectives
            # c is dominated if another point is >= c in all dims and > in at least one dim
            is_dominated = np.any(np.all(costs >= c, axis=1) & np.any(costs > c, axis=1))
            if is_dominated:
                is_efficient[i] = False
    return is_efficient


DEFAULT_BASE_PARAMS = {
    "dead_logit_bias": 0.0,
    "dead_switch_penalty": 35.0,
    "min_dead_frames": 25,
    "live_seq_weight_vis": 0.5,
    "live_seq_weight_miss": 0.45,
    "live_blend_mode": "prob",
    "live_switch_penalty": 24.0,
    "boundary_alpha": 0.7,
    "min_spell_frames": 50,
    "short_min_spell_frames": 25,
    "short_boundary_thresh": 0.5,
    "short_logit_thresh": 2.2,
    "suffix_dead_max_frames": 175,
    "suffix_dead_pD_thresh": 0.22,
    "dead_seq_weight": 0.65,
    "dead_boundary_alpha": 2.0,
    "refine_max_shift": 0,
}


def sample_configuration(rng, base_params=None):
    P = dict(base_params or DEFAULT_BASE_PARAMS)
    P["dead_logit_bias"] = float(rng.choice([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10]))
    P["dead_switch_penalty"] = float(rng.choice([25.0, 30.0, 35.0, 40.0, 45.0]))
    P["min_dead_frames"] = int(rng.choice([15, 20, 25, 30, 35]))
    P["dead_seq_weight"] = float(rng.choice([0.55, 0.60, 0.65, 0.70, 0.75]))
    P["dead_boundary_alpha"] = float(rng.choice([1.5, 2.0, 2.5, 3.0]))

    P["live_switch_penalty"] = float(rng.choice([16.0, 18.0, 20.0, 22.0, 24.0, 28.0]))
    P["boundary_alpha"] = float(rng.choice([0.6, 0.7, 0.8, 1.0]))
    P["live_seq_weight_vis"] = float(rng.choice([0.50, 0.55, 0.60, 0.65]))
    P["live_seq_weight_miss"] = 0.45
    P["live_blend_mode"] = "prob"

    P["min_spell_frames"] = 50
    P["short_min_spell_frames"] = int(rng.choice([18, 20, 22, 25]))
    P["short_boundary_thresh"] = 0.5
    P["short_logit_thresh"] = float(rng.choice([2.0, 2.2, 2.5, 2.8]))

    P["suffix_dead_max_frames"] = int(rng.choice([100, 125, 150, 175]))
    P["suffix_dead_pD_thresh"] = float(rng.choice([0.20, 0.22, 0.25, 0.28]))
    P["refine_max_shift"] = int(rng.choice([0, 6, 10, 12, 14]))
    return P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true", help="Save the best Pareto model to models/decoder_params.json")
    args = parser.parse_args()

    oof_path = os.path.join(CACHE_DIR, "metrica_oof.npz")
    print(f"Loading Metrica out-of-fold benchmark from {oof_path}...")
    bench = Bench(oof_path)

    base_path = os.path.join(MODEL_DIR, "decoder_params.json")

    rng = random.Random(args.seed)
    candidates = []

    # Include anchors
    candidates.append(("baseline_pure_metrica", dict(DEFAULT_BASE_PARAMS)))

    for i in range(args.n_samples):
        candidates.append((f"sample_{i:04d}", sample_configuration(rng, DEFAULT_BASE_PARAMS)))

    print(f"Evaluating {len(candidates)} configurations across Metrica Games 1, 2, 3...")
    t0 = time.time()
    results = []

    for idx, (name, P) in enumerate(candidates):
        s_g1, m_g1 = bench.score(P, (0,))
        s_g2, m_g2 = bench.score(P, (1,))
        s_g3, m_g3 = bench.score(P, (2,))
        s_pool, m_pool = bench.score(P, (0, 1, 2))
        worst_game = min(s_g1, s_g2, s_g3)

        results.append({
            "name": name,
            "params": P,
            "s_g1": s_g1, "m_g1": m_g1,
            "s_g2": s_g2, "m_g2": m_g2,
            "s_g3": s_g3, "m_g3": m_g3,
            "s_pool": s_pool, "m_pool": m_pool,
            "worst_game": worst_game,
        })
        if (idx + 1) % 50 == 0 or idx == len(candidates) - 1:
            print(f"[{idx+1}/{len(candidates)}] Elapsed: {time.time()-t0:.1f}s | Best Pooled: {max(r['s_pool'] for r in results):.4f}")

    # Compute Pareto frontier on (s_g1, s_g2, s_g3)
    costs = np.array([[r["s_g1"], r["s_g2"], r["s_g3"]] for r in results])
    pareto_mask = is_pareto_efficient(costs)
    pareto_candidates = [r for r, is_p in zip(results, pareto_mask) if is_p]

    print("\n" + "=" * 80)
    print(f"PARETO FRONTIER ACROSS METRICA GAMES 1, 2, 3 ({len(pareto_candidates)} solutions found)")
    print("=" * 80)
    print(f"{'Config':<20} | {'Game 1':<8} | {'Game 2':<8} | {'Game 3':<8} | {'Worst':<8} | {'Pooled sF1':<10} | {'Pooled mF1'}")
    print("-" * 80)

    # Sort Pareto candidates by pooled spellF1 descending, broken by macroF1:
    pareto_candidates.sort(key=lambda x: (x["s_pool"], x["m_pool"], x["worst_game"]), reverse=True)

    for p in pareto_candidates[:15]:
        print(f"{p['name']:<20} | {p['s_g1']:.4f}   | {p['s_g2']:.4f}   | {p['s_g3']:.4f}   | {p['worst_game']:.4f}   | {p['s_pool']:.4f}     | {p['m_pool']:.4f}")

    # Best Pareto-optimal model:
    best = pareto_candidates[0]
    print("\n" + "=" * 80)
    print("SELECTED PARETO-OPTIMAL MODEL (100% METRICA CALIBRATED):")
    print(f"Name: {best['name']}")
    print(f"Game 1 spellF1: {best['s_g1']:.4f} (macroF1: {best['m_g1']:.4f})")
    print(f"Game 2 spellF1: {best['s_g2']:.4f} (macroF1: {best['m_g2']:.4f})")
    print(f"Game 3 spellF1: {best['s_g3']:.4f} (macroF1: {best['m_g3']:.4f})")
    print(f"Worst Game:     {best['worst_game']:.4f}")
    print(f"Pooled spellF1: {best['s_pool']:.4f} (macroF1: {best['m_pool']:.4f})")
    print("=" * 80)

    final_params = dict(best["params"])
    final_params["calibrated_on"] = f"Pareto Multi-Objective Optimization strictly on Metrica Games 1-3 ({best['name'].replace('_', ' ').title()})"
    final_params["metrica_oof_spell_f1_mean"] = best["s_pool"]
    final_params["metrica_oof_macro_f1"] = best["m_pool"]

    out_file = os.path.join(MODEL_DIR, "decoder_params_metrica_pareto.json")
    with open(out_file, "w") as f:
        json.dump(final_params, f, indent=2)
    print(f"Saved Pareto parameters to {out_file}")

    if args.save_model:
        with open(base_path, "w") as f:
            json.dump(final_params, f, indent=2)
        print(f"Updated active model at {base_path}!")


if __name__ == "__main__":
    main()
