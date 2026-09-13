"""Model zoo and the native-resolution heads.

`build_model(blob)` dispatches on `blob["arch"]`. Every architecture name a
released checkpoint can carry must stay registered here or that checkpoint
stops loading.
"""

from spanmark.models.zoo import *          # noqa: F401,F403
from spanmark.models.zoo import (           # noqa: F401  explicit, the public surface
    build_model,
    Localizer,
    XLSRLocalizer,
    PositionAnchorLocalizer,
    NativePooledClassifier,
    NativePooledLocalizer,
    CoarseTemporalResidualBlock,
)
