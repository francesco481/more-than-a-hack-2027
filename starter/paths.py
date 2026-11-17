"""Resolve the delivered match assets.

The bundle as delivered does not match the layout games.py assumes: it unpacks into
data/dataset/<source>/ and the videos arrive as "<stem>-001.mp4" rather than "<stem>.mp4".
Rather than rename 19 GB of video, this module globs for whatever is actually there and
falls back to games.asset_paths() when the canonical layout is present.
"""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, SOURCES, SUFFIXES, asset_paths  # noqa: E402

CACHE = os.path.join(ROOT, "cache")

# Directories to search for a source's assets, in order of preference.
_ROOTS = (os.path.join(ROOT, "data", "dataset"), os.path.join(ROOT, "data"))


def source_dir(source="tactical"):
    for root in _ROOTS:
        folder = os.path.join(root, source)
        if os.path.isdir(folder):
            return folder
    raise IOError("no folder for source {!r} under {}".format(source, _ROOTS))


def asset(name, source="tactical"):
    """Path to one asset, or None if it is not in the bundle.

    Videos are matched by prefix so that BAR-ATH_tactical-001.mp4 resolves for "video".
    """
    folder = source_dir(source)
    stem = SOURCES[source]["stem"]
    exact = os.path.join(folder, stem + SUFFIXES[name])
    if os.path.isfile(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(folder, stem + "*" + SUFFIXES[name])))
    return matches[0] if matches else None


def need(name, source="tactical"):
    path = asset(name, source)
    if path is None:
        raise IOError("{} for source {!r} not found under {}".format(
            name, source, source_dir(source)))
    return path


def cache_path(name):
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    return os.path.join(CACHE, name)


if __name__ == "__main__":
    for source in SOURCES:
        print(source)
        for name in SOURCES[source]["assets"]:
            path = asset(name, source)
            size = "{:>12,} B".format(os.path.getsize(path)) if path else "MISSING"
            print("  {:<14} {}  {}".format(name, size, path or ""))
