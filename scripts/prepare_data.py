#!/usr/bin/env python3
"""Build manifests, and pack PartialSpoof segment labels into the layout the
evaluator reads.

    python scripts/prepare_data.py --corpus-root /path/to/PartialSpoof --split eval
    python scripts/prepare_data.py --corpus-root /path/to/PartialSpoof --split eval --labels

A manifest is `utt_id \\t n_segments \\t relpath`. `n_segments` is the number of
20 ms segments and it is authoritative — the scorer takes the count from this
column and never derives it from the audio, so a re-encoded copy cannot silently
shift the grid. That is also why this script computes the count from the file's
frame count rather than trusting a duration field.

The evaluation manifests used during development are not shipped: their
identifiers were salted and resolve against nothing in a stock corpus. Build
your own here; numbers are comparable to the reported ones only if the
underlying corpus release is the same.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

SAMPLE_RATE = 16_000
RESOLUTION_S = 0.02
HOP = int(SAMPLE_RATE * RESOLUTION_S)          # 320 samples per 20 ms segment

SPLIT_DIRS = {
    "train": "train/con_wav",
    "dev": "dev/con_wav",
    "eval": "eval/con_wav",
}
AUDIO_EXT = (".wav", ".flac")


def n_segments(path: pathlib.Path) -> int:
    """Segments on the 20 ms grid, from the true frame count."""
    import soundfile as sf
    info = sf.info(str(path))
    frames = info.frames
    if info.samplerate != SAMPLE_RATE:
        frames = round(frames * SAMPLE_RATE / info.samplerate)
    return max(1, -(-frames // HOP))           # ceil


def build_manifest(corpus_root: pathlib.Path, split: str, out: pathlib.Path) -> int:
    sub = corpus_root / SPLIT_DIRS[split]
    if not sub.is_dir():
        raise SystemExit(
            f"no {sub}\n"
            f"Expected a stock PartialSpoof layout under --corpus-root:\n"
            f"  {corpus_root}/{{train,dev,eval}}/con_wav/*.wav"
        )
    files = sorted(p for p in sub.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"{sub} has no {' or '.join(AUDIO_EXT)} files")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# utt_id\tn_segments\trelpath  (relative to SPANMARK_CORPUS_ROOT)\n")
        for i, p in enumerate(files, 1):
            fh.write(f"{p.stem}\t{n_segments(p)}\t{p.relative_to(corpus_root)}\n")
            if i % 5000 == 0:
                print(f"  {i:,}/{len(files):,}", flush=True)
    print(f"wrote {out}  ({len(files):,} utterances)")
    return len(files)


def pack_labels(corpus_root: pathlib.Path, split: str, manifest: pathlib.Path,
                out: pathlib.Path) -> None:
    """Pack per-segment labels into ids / offsets / labels (CSR-style).

    PartialSpoof ships segment labels alongside the corpus. This reads the
    2 s-resolution protocol files it publishes at the 20 ms grid; adjust
    `label_path` if your release lays them out differently.

    Polarity is **1 = bonafide, 0 = spoof** — the convention `spanmark.metrics`
    scores against. A release using the opposite convention must be inverted
    here, and nothing downstream can detect it for you: flipped labels raise no
    error, they just report roughly (100 - EER). The bonafide fraction is
    printed at the end so an inversion is visible immediately: the PartialSpoof
    eval split is 61.22 % bonafide on the 20 ms grid, so a figure near 39 %
    means your copy uses the opposite polarity and must be inverted.
    """
    label_path = corpus_root / "segment_labels" / f"{split}_seglab_0.02.npy"
    if not label_path.exists():
        raise SystemExit(
            f"no {label_path}\n"
            f"PartialSpoof publishes segment labels with the corpus. Point this at\n"
            f"the 0.02 s label file for the {split} split, or convert your copy to\n"
            f"  ids (str) / offsets (int64) / labels (int8, 1 = BONAFIDE, 0 = spoof)\n"
            f"and save it as {out}."
        )
    raw = np.load(label_path, allow_pickle=True).item()
    ids, offsets, labels, cur = [], [], [], 0
    for line in open(manifest):
        if line.startswith("#"):
            continue
        utt, n, _ = line.rstrip("\n").split("\t")
        seq = np.asarray(raw[utt], dtype=np.int8)[: int(n)]
        if len(seq) < int(n):                    # pad short label rows as bonafide
            seq = np.concatenate([seq, np.ones(int(n) - len(seq), np.int8)])
        ids.append(utt); offsets.append(cur); labels.append(seq); cur += len(seq)
    packed = np.concatenate(labels)
    bad = set(np.unique(packed)) - {0, 1}
    if bad:
        raise SystemExit(f"labels must be 0/1, saw {sorted(bad)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids, dtype=object),
                        offsets=np.array(offsets, dtype=np.int64),
                        labels=packed)
    print(f"wrote {out}  ({len(ids):,} utterances, {cur:,} segments)")
    print(f"  {100 * (packed == 1).mean():.2f} % bonafide  "
          f"(1 = bonafide, 0 = spoof; see the note in pack_labels)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("SPANMARK_CORPUS_ROOT", "data/PartialSpoof")))
    ap.add_argument("--split", choices=sorted(SPLIT_DIRS), default="eval")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--labels", action="store_true",
                    help="also pack segment labels into SPANMARK_TASKDATA")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    name = {"eval": "ps_eval", "dev": "ps_dev", "train": "ps_train"}[args.split]
    out = args.out or repo / "assets" / "manifests" / f"{name}.tsv"

    build_manifest(args.corpus_root, args.split, out)

    if args.labels:
        taskdata = pathlib.Path(os.environ.get("SPANMARK_TASKDATA", str(repo / "data" / "labels")))
        pack_labels(args.corpus_root, args.split, out, taskdata / "eval_labels" / f"{name}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
