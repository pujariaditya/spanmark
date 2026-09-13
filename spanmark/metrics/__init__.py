"""Scoring: segment EER and F1 on the resolution grid, and the score-file reader.

This is the evaluation half of the original task's grader, vendored so the repo
scores itself without depending on the harness it was developed under. The
metric definitions are unchanged — `eer` is tie-safe and matches what selected
the released checkpoint.
"""

from spanmark.metrics.metrics import auc, eer, eer_threshold, f1_at, pool_min  # noqa: F401
