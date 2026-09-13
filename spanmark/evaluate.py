#!/usr/bin/env python3
"""Score a prediction file against the labels and report segment EER.

    python -m spanmark.evaluate --scores out/ps_eval_scores.npz \
                                --manifest assets/manifests/ps_eval.tsv

Reports EER at every resolution on the grid. The 160 ms cell is the headline,
and it is the one where the two routes differ: the coarse number comes from the
native head when the prediction file carries a native grid, and from min-pooling
the 20 ms stream when it does not. Both are printed when both are available,
because the gap between them is the point of the model.

Labels are not shipped. Point SPANMARK_TASKDATA at a directory holding
`eval_labels/<dataset>.npz`; `scripts/prepare_data.py --labels` writes it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

from spanmark.metrics import eer, f1_at, eer_threshold, pool_min

RESOLUTIONS_MS = (20, 40, 80, 160, 320, 640)


def _bounds(offsets: np.ndarray, n_ids: int, total: int) -> list[int]:
    """CSR bounds from an offsets array, tolerating both conventions.

    `save_scores` and the packed label files write n+1 offsets, ending in the
    total. Some hand-built files write only the n starts. Appending the total
    unconditionally would be right for the second and silently redundant for the
    first, so decide from the length rather than assuming.
    """
    off = [int(x) for x in offsets]
    if len(off) == n_ids + 1:
        if off[-1] != total:
            raise SystemExit(
                f"offsets end at {off[-1]} but the value array is {total} long")
        return off
    if len(off) == n_ids:
        return off + [total]
    raise SystemExit(f"{len(off)} offsets for {n_ids} ids; expected {n_ids} or {n_ids + 1}")


def _taskdata() -> pathlib.Path:
    return pathlib.Path(os.environ.get(
        "SPANMARK_TASKDATA",
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "labels")))


def load_scores(path: pathlib.Path) -> tuple[dict, dict]:
    """-> (fine {utt: array}, native {utt: array} or {})."""
    with np.load(path, allow_pickle=True) as z:
        ids = [str(x) for x in z["ids"]]
        off = np.asarray(z["offsets"], dtype=np.int64)
        sc = np.asarray(z["scores"], dtype=np.float64)
        bounds = _bounds(off, len(ids), len(sc))
        fine = {u: sc[bounds[i]:bounds[i + 1]] for i, u in enumerate(ids)}
        native = {}
        if "scores_160" in z.files:
            o2 = np.asarray(z["offsets_160"], dtype=np.int64)
            s2 = np.asarray(z["scores_160"], dtype=np.float64)
            b2 = _bounds(o2, len(ids), len(s2))
            native = {u: s2[b2[i]:b2[i + 1]] for i, u in enumerate(ids)}
    return fine, native


def load_labels(dataset: str) -> dict:
    path = _taskdata() / "eval_labels" / f"{dataset}.npz"
    if not path.exists():
        raise SystemExit(
            f"no labels at {path}\n"
            f"Set SPANMARK_TASKDATA, or build them with:\n"
            f"  python scripts/prepare_data.py --split {dataset.replace('ps_','')} --labels"
        )
    with np.load(path, allow_pickle=True) as z:
        ids = [str(x) for x in z["ids"]]
        off = np.asarray(z["offsets"], dtype=np.int64)
        lab = np.asarray(z["labels"], dtype=np.int8)
        b = _bounds(off, len(ids), len(lab))
        return {u: lab[b[i]:b[i + 1]] for i, u in enumerate(ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", default="ps_eval")
    args = ap.parse_args()

    fine, native = load_scores(args.scores)
    labels = load_labels(args.dataset)

    order = [u for u in fine if u in labels]
    missing = len(fine) - len(order)
    if not order:
        raise SystemExit("no utterance in the score file has a label; wrong dataset?")
    if missing:
        print(f"note: {missing} scored utterances have no label and are skipped\n")

    print(f"{len(order):,} utterances\n")
    print(f"  {'res':>6}  {'EER':>8}  {'F1':>7}   source")
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}   {'-'*22}")

    for ms in RESOLUTIONS_MS:
        k = ms // 20
        if ms == 160 and native:
            s = np.concatenate([native[u] for u in order])
            src = "native head"
        elif k == 1:
            s = np.concatenate([fine[u] for u in order])
            src = "fine stream"
        else:
            s = np.concatenate([pool_min(fine[u], k, np.inf) for u in order])
            src = "min-pooled from 20 ms"
        y = np.concatenate([pool_min(labels[u], k, 1) for u in order])
        if len(s) != len(y):
            print(f"  {ms:>4}ms  {'--':>8}  {'--':>7}   length mismatch "
                  f"({len(s)} vs {len(y)}), skipped")
            continue
        e = eer(s, y)
        f = f1_at(s, y, eer_threshold(s, y))
        print(f"  {ms:>4}ms  {e:8.4f}  {f:7.2f}   {src}")

        # When a native grid exists, also show what pooling would have given —
        # the gap is the contribution.
        if ms == 160 and native:
            sp = np.concatenate([pool_min(fine[u], k, np.inf) for u in order])
            ep = eer(sp, y)
            # Lower EER is better, so the native head wins when ep > e. Say which
            # one won rather than printing a signed "better", which reads as
            # nonsense when the native head loses — as it does cross-corpus.
            gap = ep - e
            verdict = (f"native head wins by {gap:.4f}" if gap > 0
                       else f"POOLING wins by {-gap:.4f}")
            print(f"  {'':>6}  {ep:8.4f}  {'':>7}   min-pooled  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
