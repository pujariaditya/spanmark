"""Manifests and batching for segment-level localization.

Every utterance carries a label PER 20 ms SEGMENT, so batching pads both the
waveform and the label sequence, and a mask marks the valid segments. Nothing
is truncated: cropping a partial spoof can remove the injected span entirely
and turn a positive into a mislabelled negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
RESOLUTION_S = 0.02
HOP = int(round(RESOLUTION_S * SAMPLE_RATE))      # 320 samples


@dataclass(frozen=True)
class Utt:
    utt_id: str
    n_segments: int
    path: str          # absolute (train/dev) or relative to --eval-root (scored)


def read_manifest(path: str | Path) -> list[Utt]:
    out: list[Utt] = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 3:
            raise ValueError(f"{path}:{i}: expected 3 columns, got {len(f)}")
        out.append(Utt(f[0], int(f[1]), f[2]))
    if not out:
        raise ValueError(f"{path} is empty")
    return out


def load_label_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        ids = [str(x) for x in z["ids"]]
        off = np.asarray(z["offsets"], dtype=np.int64)
        lab = np.asarray(z["labels"], dtype=np.int8)
    return {u: lab[off[i]: off[i + 1]] for i, u in enumerate(ids)}


class SegmentDataset(Dataset):
    def __init__(self, items: list[Utt], root: str | Path = "",
                 labels: dict[str, np.ndarray] | None = None):
        self.items = items
        self.root = Path(root) if root else None
        self.labels = labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        p = Path(it.path) if self.root is None else self.root / it.path
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{p}: expected {SAMPLE_RATE} Hz, got {sr}")
        y = self.labels.get(it.utt_id) if self.labels else None
        y = torch.from_numpy(np.asarray(y, dtype=np.int64)) if y is not None else torch.zeros(0, dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(wav)), y, it.n_segments, it.utt_id


def collate(batch):
    waves, labels, nsegs, ids = zip(*batch)
    n_max = max(w.shape[0] for w in waves)
    seg_max = max(nsegs)
    wav = torch.zeros(len(waves), n_max, dtype=torch.float32)
    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long)
    lab = torch.full((len(waves), seg_max), -100, dtype=torch.long)   # -100 = ignore
    mask = torch.zeros(len(waves), seg_max, dtype=torch.bool)
    for i, (w, y, n) in enumerate(zip(waves, labels, nsegs)):
        wav[i, : w.shape[0]] = w
        mask[i, :n] = True
        if y.numel():
            lab[i, : min(n, y.numel())] = y[: min(n, y.numel())]
    return wav, lengths, lab, mask, torch.tensor(nsegs, dtype=torch.long), list(ids)
