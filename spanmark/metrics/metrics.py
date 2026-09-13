"""Localization metrics: SEGMENT-level EER and ROC-AUC.

Both take SEGMENT-level scores where HIGHER MEANS MORE BONAFIDE, and integer
labels where 1 = bonafide and 0 = spoof. Segment EER is computed over every
segment of every utterance POOLED into one flat array -- the standard
definition in the partial-spoof literature, so numbers are comparable to
published segment/frame EER at the same resolution.

EER is returned in PERCENT and AUC as a FRACTION in [0, 1] -- the same units
the HQ-MPSD paper's Table II uses (EER 0.29 %, AUC 0.998), so numbers here are
directly comparable to the published benchmark without rescaling.
"""

from __future__ import annotations

import numpy as np

__all__ = ["eer", "auc", "det_curve", "eer_threshold", "f1_at", "pool_min"]


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: scores {scores.shape} vs labels {labels.shape}"
        )
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"{n_bad} non-finite score(s) (NaN/inf)")
    bad = set(np.unique(labels)) - {0, 1}
    if bad:
        raise ValueError(f"labels must be 0/1, saw {sorted(bad)}")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"needs both classes present (bonafide={n_pos}, spoof={n_neg})"
        )
    return scores, labels, n_pos, n_neg


def det_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frr, far, thresholds), evaluated only at DISTINCT score values.

    The widely-copied ASVspoof reference implementation instead ranks every
    trial individually, which silently breaks ties in whatever order the target
    and non-target arrays were concatenated. That makes an all-constant scorer
    come out at EER 100 % rather than chance, and lets any saturated model drift
    by a few tenths of a point purely from utterance ordering. Evaluating per
    distinct threshold removes the ordering dependency and reproduces sklearn's
    roc_curve exactly. With continuous scores and no ties the two are identical,
    so ASVspoof-style published numbers stay directly comparable.
    """
    scores, labels, n_pos, n_neg = _check(scores, labels)

    # descending: threshold t accepts everything scoring >= t
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    # keep only the last index of each run of equal scores
    distinct = np.r_[np.nonzero(np.diff(s))[0], np.array([s.size - 1])]
    tps, fps, thr = tps[distinct], fps[distinct], s[distinct]

    # prepend the "accept nothing" operating point: FAR 0, FRR 1
    far = np.concatenate(([0.0], fps / n_neg))
    frr = np.concatenate(([1.0], 1.0 - tps / n_pos))
    thresholds = np.concatenate(([np.inf], thr))
    return frr, far, thresholds


def _eer_index(scores: np.ndarray, labels: np.ndarray):
    """(frr, far, thresholds, idx) at the operating point closest to FRR == FAR.

    One implementation shared by eer() and eer_threshold() so the EER and the
    F1 reported beside it can never drift onto different operating points.
    """
    frr, far, thr = det_curve(scores, labels)
    idx = int(np.nanargmin(np.abs(frr - far)))
    return frr, far, thr, idx


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate in PERCENT. Higher score = more bonafide."""
    frr, far, _, idx = _eer_index(scores, labels)
    return float(np.mean((frr[idx], far[idx])) * 100.0)


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The score threshold at the EER operating point: accept >= tau as bonafide.

    det_curve() prepends an "accept nothing" point whose threshold is +inf. That
    is a legitimate EER operating point only in degenerate cases, but it is never
    a usable decision threshold, so fall back to the largest finite score.
    """
    _, _, thr, idx = _eer_index(scores, labels)
    tau = float(thr[idx])
    if not np.isfinite(tau):
        finite = thr[np.isfinite(thr)]
        tau = float(finite.max()) if finite.size else 0.0
    return tau


def f1_at(
    scores: np.ndarray, labels: np.ndarray, tau: float, positive: str = "bona"
) -> float:
    """F1 in PERCENT at threshold `tau`, for `positive` in {"bona", "spoof"}.

    The decision rule is the one det_curve() assumes: score >= tau is called
    bonafide. So the bonafide-positive prediction is (score >= tau) and the
    spoof-positive prediction is its complement.

    Which class is positive matters a great deal here: spoof prevalence runs
    from 38.8 % at 20 ms to 73.6 % at 640 ms once labels are min-pooled, so the
    two conventions diverge by more than two points at both ends of the sweep.
    The published F1 column this task is scored against is bonafide-positive.
    """
    if positive not in ("bona", "spoof"):
        raise ValueError(f"positive must be 'bona' or 'spoof', got {positive!r}")
    scores, labels, _, _ = _check(scores, labels)

    accept = scores >= tau
    if positive == "bona":
        pred, truth = accept, labels == 1
    else:
        pred, truth = ~accept, labels == 0

    tp = float(np.count_nonzero(pred & truth))
    fp = float(np.count_nonzero(pred & ~truth))
    fn = float(np.count_nonzero(~pred & truth))
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:          # no predictions and no truths for this class
        return 0.0
    return float(200.0 * tp / denom)


def pool_min(x: np.ndarray, k: int, pad_value: float) -> np.ndarray:
    """Min-pool `x` into ceil(len(x)/k) blocks of k, padding the tail.

    This is exactly how PartialSpoof derives its 40..640 ms segment labels from
    the 20 ms grid: a block is spoof iff any of its sub-segments is spoof, and
    with 1 = bonafide that is a min. Verified bit-identical against the corpus's
    own *_seglab_*.npy files at 0.04 / 0.16 / 0.64 s.

    Labels pad with 1 (bonafide, so padding cannot make a block spoof); scores
    pad with +inf (so padding can never win the min and set the block score).
    Pool per utterance -- pooling a concatenated array would blend blocks across
    utterance boundaries.
    """
    if k < 1:
        raise ValueError(f"block factor must be >= 1, got {k}")
    x = np.asarray(x).ravel()
    if k == 1:
        return x
    n = -(-x.size // k)                       # ceil division
    pad = n * k - x.size
    if pad:
        x = np.concatenate([x, np.full(pad, pad_value, dtype=x.dtype)])
    return x.reshape(n, k).min(axis=1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC as a FRACTION in [0, 1], via the Mann-Whitney U rank statistic.

    Ties get average ranks, making this identical to sklearn's roc_auc_score
    (including the half credit a tied pair earns) with no sklearn dependency.
    """
    scores, labels, n_pos, n_neg = _check(scores, labels)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    # average ranks within each run of equal scores (1-based)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
