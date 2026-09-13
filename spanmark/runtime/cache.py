"""Validated, full-utterance caches for native-resolution training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StateCache:
    """Memory-mapped second-pass states with their exact utterance partition."""

    path: Path
    metadata: dict
    ids: np.ndarray
    offsets: np.ndarray
    states: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    @classmethod
    def open(cls, path: str | Path) -> "StateCache":
        path = Path(path)
        if not (path / "COMPLETE").exists():
            raise ValueError(f"cache is not marked complete: {path}")
        metadata = json.loads((path / "metadata.json").read_text())
        ids = np.load(path / "ids.npy", allow_pickle=False)
        offsets = np.load(path / "offsets.npy", allow_pickle=False)
        states = np.load(path / "states.npy", allow_pickle=False, mmap_mode="r")
        scores = np.load(path / "scores.npy", allow_pickle=False, mmap_mode="r")
        labels = np.load(path / "labels.npy", allow_pickle=False, mmap_mode="r")

        utterances = int(metadata["utterances"])
        segments = int(metadata["segments"])
        state_dim = int(metadata["state_dim"])
        if ids.shape != (utterances,) or offsets.shape != (utterances + 1,):
            raise ValueError("cache id/offset shapes disagree with metadata")
        if states.shape != (segments, state_dim):
            raise ValueError("cache state shape disagrees with metadata")
        if scores.shape != (segments,) or labels.shape != (segments,):
            raise ValueError("cache score/label shape disagrees with metadata")
        if offsets[0] != 0 or offsets[-1] != segments or np.any(np.diff(offsets) <= 0):
            raise ValueError("cache offsets are not a strictly increasing partition")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("cache labels must be binary")
        if not np.isfinite(scores).all():
            raise ValueError("cache contains non-finite fine scores")
        return cls(path, metadata, ids, offsets, states, scores, labels)
