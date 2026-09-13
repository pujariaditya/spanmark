"""Private ground truth for PS/LPS localization.

    eval_labels/<ds>.npz    ids + offsets + labels, 20 ms grid, 1=bonafide 0=spoof
    train_labels/<ds>.npz   same shape, for the public train/dev splits
    meta.json               counts and provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "DATASETS", "RESOLUTION_S", "RESOLUTIONS_MS", "BLOCKS",
    "load_labels", "load_meta",
]

DATASETS = ("ps_eval", "lps")
RESOLUTION_S = 0.02

# The scored resolution sweep. Labels are stored once, on the 20 ms grid; the
# coarser grids are derived by min-pooling (a block is spoof iff any 20 ms
# sub-segment is spoof). That derivation is bit-identical to PartialSpoof's own
# *_seglab_*.npy files -- verified at 0.04 / 0.16 / 0.64 s, zero mismatches over
# the full 71,237-utterance eval set -- so the private store stays 20 ms-only.
RESOLUTIONS_MS = (20, 40, 80, 160, 320, 640)
BLOCKS = tuple(ms // 20 for ms in RESOLUTIONS_MS)          # (1, 2, 4, 8, 16, 32)


def _root(private_dir: str | Path) -> Path:
    base = Path(private_dir)
    nested = base / "taskdata"
    return nested if (nested / "eval_labels").is_dir() else base


def load_labels(private_dir: str | Path, ds: str) -> dict[str, np.ndarray]:
    path = _root(private_dir) / "eval_labels" / f"{ds}.npz"
    if path.is_symlink():
        raise RuntimeError(f"refusing to read a symlinked label file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"missing ground truth for {ds}: {path}")
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        off = np.asarray(z["offsets"], dtype=np.int64)
        lab = np.asarray(z["labels"], dtype=np.int8)
    return {u: lab[off[i] : off[i + 1]] for i, u in enumerate(ids)}


def load_meta(private_dir: str | Path) -> dict:
    p = _root(private_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}
