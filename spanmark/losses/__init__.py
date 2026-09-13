"""Training-only objectives. None of these run at inference."""

from spanmark.losses.risk import (          # noqa: F401
    native_risk_loss,
    pairwise_ranking_loss,
    smooth_eer_loss,
)
