"""Turn the delivered match assets into memory-mappable arrays the pipeline can open instantly.

The ATD text file is 195 MB of ASCII and parsing it takes the better part of a minute.
This script parses it once into .npy files in cache/ that numpy can mmap instantly.

    python starter/prep.py                 # tactical (what a submission is indexed to)
    python starter/prep.py --source broadcast
    python starter/prep.py --all

Every ATD line is exactly "<frame>:<46 x x,y,s>:<ball x,y,s>", verified rectangular on the
delivered file, so the whole chunk parses in one numpy call rather than field by field.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

try:
    from starter import paths  # noqa: E402
    from starter.paths import ROOT, GAME, GAMES, SOURCES  # noqa: E402
except ImportError:
    import paths  # noqa: E402
    from paths import ROOT, GAME, GAMES, SOURCES  # noqa: E402

# Params
CHUNK_LINES = 20000          # lines parsed per numpy call; memory/speed trade-off only
HOME, AWAY = 0, 1
MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]


def slot_teams(source):
    """(n_slots,) int8 of HOME/AWAY per track slot, plus the two club names.

    The first <Team> in the FIFA/EPTS metadata is the home side -- the convention kloppy and
    starter/baseline.py both follow. For this match that is Barcelona, which DATA.md states
    independently, so the two agree.
    """
    root = ET.parse(paths.need("atd_meta", source)).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(teams)))
    home_id = teams[0][0]
    codes = np.array([HOME if player.get("teamId") == home_id else AWAY
                      for player in root.iter("Player")], dtype=np.int8)
    return codes, (teams[0][1], teams[1][1])


def parse_atd(source):
    """(players (n, slots, 3), ball (n, 3)) float32, indexed by VIDEO frame.

    Channels are x, y, speed: x and y normalised 0-1 over the pitch, speed in m/s, NaN where
    the tracker saw nothing. frame_id counts from 1 and a submission row counts from 0, so the
    arrays are shifted down by one on write -- row i is video frame i, everywhere in the pipeline.
    """
    path = paths.need("atd_raw", source)
    n_frames = SOURCES[source]["n_video_frames"]
    n_slots = len(slot_teams(source)[0])
    stride = 1 + 3 * n_slots + 3

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    seen = 0
    with open(path) as handle:
        while True:
            lines = handle.readlines(CHUNK_LINES * 160)
            if not lines:
                break
            # Every separator -- field, group and line -- becomes a comma so that one
            # numpy call parses the whole chunk. The file is rectangular, so the reshape below
            # is the check that it really was.
            flat = "".join(lines).replace(";", ",").replace(":", ",").replace("\n", ",")
            values = np.fromstring(flat.rstrip(","), dtype=np.float64, sep=",")
            if values.size % stride:
                raise ValueError("{}: chunk is not {} fields per line".format(path, stride))
            table = values.reshape(-1, stride)
            rows = table[:, 0].astype(np.int64) - 1          # frame_id -> video frame
            players[rows] = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
            ball[rows] = table[:, 1 + 3 * n_slots:]
            seen += len(table)
            print("\r  {:>7,} / {:,} frames".format(seen, n_frames), end="", flush=True)
    print()
    if seen != n_frames:
        print("  WARNING: {} records for {} video frames".format(seen, n_frames))
    return players, ball


def parse_homography(source):
    """(matrices (n, 3, 3) float32, reprojection_error (n,) float32, valid (n,) bool)."""
    table = pd.read_csv(paths.need("homography", source))
    matrices = table[MATRIX_COLUMNS].values.astype(np.float32).reshape(-1, 3, 3)
    return (matrices,
            table["reprojection_error"].values.astype(np.float32),
            table["valid"].values.astype(bool))


def parse_smart_tagging(fps=25):
    """Smart Tagging instances as video-frame intervals.

    <start>/<end> are seconds from the first frame of the tactical video -- verified by
    starter/load_smart_tagging.py against both kickoffs -- so round(seconds * fps) is a frame
    and a submission row directly.
    """
    path = paths.asset("smart_tagging", "tactical")
    if path is None:
        return []
    out = []
    for index, instance in enumerate(ET.parse(path).iter("instance")):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "id": index,
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return sorted(out, key=lambda event: event["start_frame"])


def build(source):
    print("[{}]".format(source))
    codes, names = slot_teams(source)
    print("  teams: H = {}, A = {}   slots: {} home / {} away".format(
        names[0], names[1], int((codes == HOME).sum()), int((codes == AWAY).sum())))
    np.save(paths.cache_path("{}_slot_team.npy".format(source)), codes)

    print("  parsing ATD ...")
    players, ball = parse_atd(source)
    np.save(paths.cache_path("{}_players.npy".format(source)), players)
    np.save(paths.cache_path("{}_ball.npy".format(source)), ball)
    visible = np.isfinite(ball[:, 0])
    print("  ball visible on {:,} / {:,} frames ({:.1f}%)".format(
        int(visible.sum()), len(ball), 100.0 * visible.mean()))

    print("  parsing homography ...")
    matrices, error, valid = parse_homography(source)
    np.save(paths.cache_path("{}_homography.npy".format(source)), matrices)
    np.save(paths.cache_path("{}_reproj.npy".format(source)), error)
    np.save(paths.cache_path("{}_valid.npy".format(source)), valid)
    print("  homography valid on {:.1f}% of frames".format(100.0 * valid.mean()))

    meta = {
        "source": source,
        "game": GAME,
        "n_frames": int(len(ball)),
        "fps": GAMES[GAME]["fps"],
        "pitch": list(GAMES[GAME]["pitch"]),
        "n_slots": int(len(codes)),
        "team_names": list(names),
        "ball_visible_fraction": float(visible.mean()),
        "video": paths.asset("video", source),
    }
    with open(paths.cache_path("{}_meta.json".format(source)), "w") as handle:
        json.dump(meta, handle, indent=2)

    if source == "tactical":
        events = parse_smart_tagging(meta["fps"])
        with open(paths.cache_path("smart_tagging.json"), "w") as handle:
            json.dump(events, handle, indent=1)
        print("  smart tagging: {} instances".format(len(events)))
    print("  done\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=sorted(SOURCES))
    parser.add_argument("--all", action="store_true", help="both recordings")
    args = parser.parse_args()
    for source in (sorted(SOURCES) if args.all else [args.source]):
        build(source)
    print("cache: {}".format(paths.CACHE))
