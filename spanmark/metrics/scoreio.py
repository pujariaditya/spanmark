"""Grader-side reader for ragged per-segment score files.

Parses the .npz directly and never imports the agent's coral_io: that module
lives in the submission and could otherwise decide how its own output is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_scores", "read_native_scores", "ScoreFileError"]


class ScoreFileError(RuntimeError):
    """The score file is missing, malformed, or unreadable."""


def _unpack(ids: np.ndarray, off: np.ndarray, val: np.ndarray, what: str) -> dict:
    if off.size != ids.size + 1:
        raise ScoreFileError(f"{what}: offsets has {off.size} entries, expected {ids.size + 1}")
    if off[0] != 0 or off[-1] != val.size:
        raise ScoreFileError(
            f"{what}: offsets must span the score array: got [{off[0]}, {off[-1]}] "
            f"for {val.size} scores"
        )
    if np.any(np.diff(off) < 0):
        raise ScoreFileError(f"{what}: offsets must be non-decreasing")
    out = {str(u): val[off[i] : off[i + 1]] for i, u in enumerate(ids.tolist())}
    if len(out) != ids.size:
        raise ScoreFileError(
            f"{what}: duplicate utterance ids ({ids.size} rows, {len(out)} unique)"
        )
    return out


def read_scores(path: str | Path) -> dict[str, np.ndarray]:
    """The mandatory 20 ms grid: keys ids / offsets / scores."""
    path = Path(path)
    if not path.exists():
        raise ScoreFileError(f"score file was never written: {path}")
    if path.stat().st_size == 0:
        raise ScoreFileError(f"score file is empty (0 bytes): {path}")
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = {"ids", "offsets", "scores"} - set(z.files)
            if missing:
                raise ScoreFileError(
                    f"missing array(s) {sorted(missing)}; found {sorted(z.files)}. "
                    "Write with coral_io.save_scores() -- localization needs "
                    "ids/offsets/scores, not a flat per-utterance array."
                )
            ids = np.asarray(z["ids"]).ravel()
            off = np.asarray(z["offsets"], dtype=np.int64).ravel()
            val = np.asarray(z["scores"], dtype=np.float64).ravel()
    except ScoreFileError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScoreFileError(f"could not read {path.name} as .npz: {exc}") from exc
    return _unpack(ids, off, val, "20 ms grid")


def read_native_scores(path: str | Path, resolutions_ms) -> dict[int, dict]:
    """OPTIONAL per-resolution grids: keys offsets_<ms> / scores_<ms>.

    A model may predict natively at a coarse resolution instead of having its
    20 ms output min-pooled. That is how the published systems this task is
    scored against actually work, and pooling a fine output is a strictly
    weaker way to reach 640 ms. Any resolution left out is pooled as before, so
    a submission that writes only the 20 ms grid keeps working unchanged.

    `ids` is shared with the 20 ms grid -- the same utterances in the same
    order -- so only the offsets and values are per-resolution.
    """
    path = Path(path)
    out: dict[int, dict] = {}
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            ids = np.asarray(z["ids"]).ravel()
            for ms in resolutions_ms:
                ok, sk = f"offsets_{ms}", f"scores_{ms}"
                if ok not in files and sk not in files:
                    continue
                if ok not in files or sk not in files:
                    raise ScoreFileError(
                        f"{ms} ms: need BOTH {ok} and {sk}; found only "
                        f"{ok if ok in files else sk}"
                    )
                off = np.asarray(z[ok], dtype=np.int64).ravel()
                val = np.asarray(z[sk], dtype=np.float64).ravel()
                out[ms] = _unpack(ids, off, val, f"{ms} ms grid")
    except ScoreFileError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScoreFileError(f"could not read per-resolution grids: {exc}") from exc
    return out
