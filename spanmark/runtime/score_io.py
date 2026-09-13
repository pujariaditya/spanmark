"""Score I/O for partial-spoof LOCALIZATION.

Localization emits ONE SCORE PER SEGMENT, and utterances have different
lengths, so the payload is ragged. It is stored CSR-style: a flat score array
plus offsets, which round-trips exactly and needs no pickling.

    ids      (N,)    unicode utterance ids
    offsets  (N+1,)  int64, scores for utterance i are scores[off[i]:off[i+1]]
    scores   (T,)    float32, higher = MORE BONAFIDE

Optional native coarse grids share ``ids`` and add ``offsets_<ms>`` plus
``scores_<ms>``. Their per-utterance count is exactly ceil(n_20ms / (ms/20)).

The grader parses this itself and never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["save_scores", "load_scores"]

RESOLUTIONS_MS = (20, 40, 80, 160, 320, 640)


def save_scores(path: str | Path, scores: Mapping[str, Sequence[float]],
                native: Mapping[int, Mapping[str, Sequence[float]]] | None = None) -> None:
    """Write mandatory 20 ms scores and optional native coarse grids.

    Emit the LOGIT DIFFERENCE per segment, logit(bonafide) - logit(spoof), so
    0 is the decision boundary and the ranking keeps its resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(scores.keys())
    if not ids:
        raise ValueError("refusing to write an empty score file")

    lengths, flat = [], []
    for utt in ids:
        arr = np.asarray(scores[utt], dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError(f"{utt}: empty score array")
        lengths.append(arr.size)
        flat.append(arr)
    values = np.concatenate(flat)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{int((~np.isfinite(values)).sum())} non-finite score(s); the grader rejects these"
        )
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    arrays = {"ids": np.asarray(ids, dtype=np.str_),
              "offsets": offsets, "scores": values}

    # Optional native coarse grids replace min-pooling only at the resolutions
    # supplied here. The mandatory 20 ms lengths define every expected count.
    for ms, per_utt in (native or {}).items():
        if ms not in RESOLUTIONS_MS or ms == 20:
            raise ValueError(
                f"native grid must be one of {RESOLUTIONS_MS[1:]}, got {ms}"
            )
        extras = set(per_utt) - set(ids)
        if extras:
            raise ValueError(
                f"{ms} ms grid has unknown utterance(s): {sorted(extras)[:3]}"
            )
        factor = ms // 20
        native_lengths, chunks = [], []
        for i, utt in enumerate(ids):
            if utt not in per_utt:
                raise ValueError(f"{ms} ms grid is missing utterance {utt}")
            arr = np.asarray(per_utt[utt], dtype=np.float32).ravel()
            want = -(-int(lengths[i]) // factor)
            if arr.size != want:
                raise ValueError(
                    f"{utt}: {ms} ms grid has {arr.size} blocks, expected {want} "
                    f"= ceil({lengths[i]} / {factor})"
                )
            native_lengths.append(arr.size)
            chunks.append(arr)
        values_native = np.concatenate(chunks)
        if not np.isfinite(values_native).all():
            raise ValueError(f"{ms} ms grid has non-finite scores")
        offsets_native = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(native_lengths, out=offsets_native[1:])
        arrays[f"offsets_{ms}"] = offsets_native
        arrays[f"scores_{ms}"] = values_native

    np.savez_compressed(path, **arrays)


def load_scores(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        off = np.asarray(z["offsets"], dtype=np.int64)
        val = np.asarray(z["scores"], dtype=np.float32)
    return {u: val[off[i] : off[i + 1]] for i, u in enumerate(ids)}
