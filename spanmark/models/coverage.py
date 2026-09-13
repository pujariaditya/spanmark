"""Ordinal spoof-coverage refinement for a frozen duration-native160 model.

The inherited event field already normalizes all nonempty eight-frame spoof
patterns.  This module collapses that distribution by spoof-frame count and
uses the resulting bounded statistics in one trained scalar residual.  The
zero residual is exactly the inherited segmental score; no model outputs are
averaged or post-processed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spanmark.models.segmental import NativeSegmentalClassifier, NativeSegmentalLocalizer


_ALIGNED_COUNT_FREQUENCIES = (
    254930, 30705, 21247, 11931, 9952, 9531, 9490, 9129, 201492,
)
_COVERAGE_KINDS = ("binary", "ordinal_uniform", "ordinal_count_balanced")


def event_count_probabilities(
    pattern_energies: torch.Tensor,
    log_z_any: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Collapse conditional nonempty-pattern mass into spoof counts 1..8.

    ``pattern_energies`` ends in the 256 bit patterns used by the event parent.
    Only indices ``1 .. 2**length - 1`` are supported.  Empty blocks return an
    all-zero vector; every nonempty block returns probabilities summing to one.
    """
    if (pattern_energies.ndim < 1 or pattern_energies.shape[-1] != 256
            or not pattern_energies.is_floating_point()):
        raise ValueError("pattern energies must be floating point with 256 states")
    expected = pattern_energies.shape[:-1]
    if log_z_any.shape != expected or lengths.shape != expected:
        raise ValueError("event partition and length shapes must match energies")
    if log_z_any.device != pattern_energies.device or lengths.device != pattern_energies.device:
        raise ValueError("event count inputs must share a device")
    if bool((lengths < 0).any()) or bool((lengths > 8).any()):
        raise ValueError("event lengths must lie in [0,8]")

    indices = torch.arange(256, device=pattern_energies.device)
    supported = (
        (indices > 0)
        & (indices < (1 << lengths.unsqueeze(-1)))
    )
    log_probability = pattern_energies - log_z_any.unsqueeze(-1)
    probability = torch.exp(log_probability.masked_fill(~supported, -torch.inf))
    bit_count = sum((indices >> slot) & 1 for slot in range(8))
    bucket = (bit_count.clamp_min(1) - 1).expand_as(probability)
    result = probability.new_zeros(expected + (8,))
    result.scatter_add_(-1, bucket, probability)
    return result


def _ordinal_initial_thresholds(frequencies: tuple[int, ...]) -> list[float]:
    """Empirical-prior log-odds shifts, with the any-spoof threshold at zero."""
    total = float(sum(frequencies))
    exceedance = [sum(frequencies[k + 1 :]) / total for k in range(8)]

    def logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    base = logit(exceedance[0])
    return [base - logit(probability) for probability in exceedance]


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class NativeCoverageClassifier(NativeSegmentalClassifier):
    """Frozen segmental parent plus one count-distribution score residual."""

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160,),
        *args,
        coverage_kind: str = "binary",
        coverage_count_frequencies: tuple[int, ...] = _ALIGNED_COUNT_FREQUENCIES,
        **kwargs,
    ):
        if coverage_kind not in _COVERAGE_KINDS:
            raise ValueError(f"unknown coverage objective: {coverage_kind}")
        frequencies = tuple(int(value) for value in coverage_count_frequencies)
        if len(frequencies) != 9 or any(value <= 0 for value in frequencies):
            raise ValueError("coverage count frequencies must contain nine positives")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.coverage_kind = coverage_kind
        self.coverage_count_frequencies = frequencies
        self.coverage_residual = nn.Linear(9, 1)
        nn.init.zeros_(self.coverage_residual.weight)
        nn.init.zeros_(self.coverage_residual.bias)
        if coverage_kind != "binary":
            thresholds = _ordinal_initial_thresholds(frequencies)
            gaps = [right - left for left, right in zip(thresholds[:-1], thresholds[1:])]
            self.coverage_threshold_raw = nn.Parameter(torch.tensor(
                [_inverse_softplus(gap) for gap in gaps], dtype=torch.float32,
            ))
        self._coverage_aux: dict[str, torch.Tensor] = {}

    def coverage_thresholds(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(1)
        if self.coverage_kind == "binary":
            return zero
        gaps = F.softplus(self.coverage_threshold_raw).to(reference.dtype)
        return torch.cat([zero, gaps.cumsum(dim=0)])

    def count_weights(self, reference: torch.Tensor) -> torch.Tensor:
        frequencies = reference.new_tensor(self.coverage_count_frequencies)
        weights = frequencies.rsqrt()
        normalization = (frequencies * weights).sum() / frequencies.sum()
        return weights / normalization

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._coverage_aux = {}
        return super().forward(states, valid)

    def _refine_native_score(
        self,
        ms: int,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        parent_score = super()._refine_native_score(
            ms, block_states, block_mask, score
        )
        pattern_probability = event_count_probabilities(
            self._event_aux["pattern_energies"],
            self._event_aux["logZ_any"],
            self._event_aux["lengths"],
        )
        length_feature = self._event_aux["lengths"].to(
            pattern_probability.dtype
        ).unsqueeze(-1) / 8.0
        features = torch.cat([pattern_probability, length_feature], dim=-1)
        delta = self.coverage_residual(features).squeeze(-1)
        block_valid = block_mask.any(dim=-1)
        spoof_latent = -parent_score + delta
        thresholds = self.coverage_thresholds(spoof_latent)
        self._coverage_aux.update(
            parent_score=parent_score,
            count_probabilities=pattern_probability,
            features=features,
            delta=delta,
            spoof_latent=spoof_latent,
            thresholds=thresholds,
            block_valid=block_valid,
        )
        return (parent_score - delta).masked_fill(~block_valid, 0.0)

    def coverage_loss(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """Binary or ordered exceedance loss on exact spoof-frame counts."""
        if "spoof_latent" not in self._coverage_aux:
            raise RuntimeError("call the coverage classifier before coverage_loss")
        previous_valid = self._event_aux.get("frame_valid")
        if (previous_valid is None or frame_labels.shape != previous_valid.shape
                or valid.shape != previous_valid.shape):
            raise ValueError("coverage labels must match the preceding forward")
        if (valid.dtype != torch.bool or valid.device != previous_valid.device
                or frame_labels.device != previous_valid.device
                or not torch.equal(valid, previous_valid)):
            raise ValueError("coverage validity must match the preceding forward")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid coverage frame labels must be binary")

        spoof = (frame_labels == 0) & valid
        count = self._blockify(spoof, 8, False).sum(dim=-1).long()
        block_valid = self._blockify(valid, 8, False).any(dim=-1)
        if not torch.equal(block_valid, self._coverage_aux["block_valid"]):
            raise ValueError("coverage labels differ from the preceding forward")
        latent = self._coverage_aux["spoof_latent"]
        thresholds = self._coverage_aux["thresholds"]
        logits = latent.unsqueeze(-1) - thresholds
        targets = count.unsqueeze(-1) > torch.arange(
            thresholds.numel(), device=count.device
        )
        per_task = F.binary_cross_entropy_with_logits(
            logits, targets.to(logits.dtype), reduction="none"
        )
        per_block = per_task.mean(dim=-1)
        if self.coverage_kind == "ordinal_count_balanced":
            per_block = per_block * self.count_weights(per_block)[count]
        self._coverage_aux.update(
            spoof_count=count, ordinal_logits=logits, ordinal_targets=targets,
        )
        return per_block[block_valid].mean()


class NativeCoverageLocalizer(NativeSegmentalLocalizer):
    """Inherited XLS-R reader with one ordinal-coverage native160 model."""

    def __init__(
        self,
        *args,
        native_resolutions_ms: tuple[int, ...] = (160,),
        native_attention_hidden: int = 64,
        native_classifier_hidden: int = 128,
        native_dropout: float = 0.1,
        native_trunk_layers: int = 0,
        native_trunk_bottleneck: int = 64,
        native_zero_output_init: bool = True,
        native_summary_mode: str = "meanstd",
        native_block_layers: int = 0,
        native_block_bottleneck: int = 64,
        native_event_hidden: int = 64,
        native_event_pairwise: bool = True,
        native_segmental_use_duration: bool = False,
        native_segmental_use_context_boundary: bool = False,
        native_segmental_boundary_hidden: int = 32,
        native_segmental_duration_bins: tuple[int, ...] = NativeSegmentalClassifier._DURATION_BINS,
        native_coverage_kind: str = "binary",
        native_coverage_count_frequencies: tuple[int, ...] = _ALIGNED_COUNT_FREQUENCIES,
        **kwargs,
    ):
        parent_args = dict(
            native_resolutions_ms=tuple(native_resolutions_ms),
            native_attention_hidden=native_attention_hidden,
            native_classifier_hidden=native_classifier_hidden,
            native_dropout=native_dropout,
            native_trunk_layers=native_trunk_layers,
            native_trunk_bottleneck=native_trunk_bottleneck,
            native_zero_output_init=native_zero_output_init,
            native_summary_mode=native_summary_mode,
            native_block_layers=native_block_layers,
            native_block_bottleneck=native_block_bottleneck,
            native_event_hidden=native_event_hidden,
            native_event_pairwise=native_event_pairwise,
            native_segmental_use_duration=native_segmental_use_duration,
            native_segmental_use_context_boundary=native_segmental_use_context_boundary,
            native_segmental_boundary_hidden=native_segmental_boundary_hidden,
            native_segmental_duration_bins=tuple(native_segmental_duration_bins),
        )
        super().__init__(*args, **parent_args, **kwargs)
        inherited = self.native_pool.state_dict()
        pool_args = {
            key.removeprefix("native_"): value for key, value in parent_args.items()
        }
        self.native_pool = NativeCoverageClassifier(
            self.out2.in_features,
            **pool_args,
            coverage_kind=native_coverage_kind,
            coverage_count_frequencies=tuple(native_coverage_count_frequencies),
        )
        missing, unexpected = self.native_pool.load_state_dict(inherited, strict=False)
        if unexpected or any(not name.startswith("coverage_") for name in missing):
            raise RuntimeError(f"coverage localizer transfer failed: {missing}/{unexpected}")
