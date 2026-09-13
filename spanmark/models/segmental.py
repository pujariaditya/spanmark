"""Exact maximal-run inference across native 160 ms blocks.

The frozen event-field parent supplies one logit per aligned native block.  A
binary semi-Markov CRF then normalizes complete alternating spoof/bonafide runs
over the utterance.  Zero structural potentials reproduce the parent block
logits exactly; learned run, duration, and optional boundary potentials are one
trained native model, not post-processing or an ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spanmark.models.event import NativeEventClassifier, NativeEventLocalizer


_FLIPPED_BINARY = (1, 0)


def _check_segmental_inputs(
    unary: torch.Tensor,
    lengths: torch.Tensor,
    run_potential: torch.Tensor,
    start_potential: torch.Tensor,
    duration_potential: torch.Tensor | None,
    boundary_potential: torch.Tensor | None,
) -> tuple[int, int]:
    if unary.ndim != 3 or unary.shape[-1] != 2 or not unary.is_floating_point():
        raise ValueError("segmental unary potentials must have shape (B,T,2)")
    batch, steps, _ = unary.shape
    if tuple(lengths.shape) != (batch,):
        raise ValueError(f"segmental lengths must have shape ({batch},)")
    if lengths.device != unary.device:
        raise ValueError("segmental lengths and potentials must share a device")
    if steps < 1 or bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError("every segmental sequence length must be in [1,T]")
    for name, value in (
        ("run", run_potential), ("start", start_potential),
    ):
        if value.shape != (2,) or not value.is_floating_point():
            raise ValueError(f"{name} potential must have shape (2,)")
        if value.device != unary.device:
            raise ValueError(f"{name} potential must share the unary device")
    if duration_potential is not None:
        if (duration_potential.ndim != 2 or duration_potential.shape[0] != 2
                or duration_potential.shape[1] < 1
                or not duration_potential.is_floating_point()):
            raise ValueError("duration potential must have shape (2,K), K>=1")
        if duration_potential.device != unary.device:
            raise ValueError("duration potential must share the unary device")
    if boundary_potential is not None:
        if (boundary_potential.shape != unary.shape
                or not boundary_potential.is_floating_point()):
            raise ValueError("boundary potential must match unary shape")
        if boundary_potential.device != unary.device:
            raise ValueError("boundary potential must share the unary device")
    return batch, steps


def _duration_values(
    duration_potential: torch.Tensor | None,
    durations: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return (candidate-duration, label) power-of-two bucket potentials."""
    if duration_potential is None:
        return reference.new_zeros((durations.numel(), 2))
    buckets = torch.floor(torch.log2(durations.float())).long()
    buckets = buckets.clamp_max(duration_potential.shape[1] - 1)
    return duration_potential.index_select(1, buckets).transpose(0, 1)


def maximal_run_log_marginals(
    unary: torch.Tensor,
    lengths: torch.Tensor,
    run_potential: torch.Tensor,
    start_potential: torch.Tensor,
    duration_potential: torch.Tensor | None = None,
    boundary_potential: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact block-label log marginals for an alternating binary semi-CRF.

    Segments have positive length, cover the valid prefix, and adjacent segments
    must have opposite labels.  Therefore every binary label sequence maps to
    exactly one segmentation into maximal constant-label runs.  The returned
    tensor has shape ``(B,T,2)`` and invalid padding is zeroed.
    """
    batch, steps = _check_segmental_inputs(
        unary, lengths, run_potential, start_potential,
        duration_potential, boundary_potential,
    )
    device = unary.device
    flipped = torch.tensor(_FLIPPED_BINARY, device=device)
    positions = torch.arange(steps, device=device)
    valid = positions[None, :] < lengths[:, None]
    unary = unary.masked_fill(~valid.unsqueeze(-1), 0.0)
    prefix = torch.cat(
        [unary.new_zeros((batch, 1, 2)), unary.cumsum(dim=1)], dim=1
    )

    # alpha[e,y] sums segmentations of [0,e) ending in a y segment.
    alpha: list[torch.Tensor] = [unary.new_zeros((batch, 2))]
    for end in range(1, steps + 1):
        starts = torch.arange(end - 1, -1, -1, device=device)
        durations = end - starts
        previous = torch.stack(alpha, dim=1).index_select(1, starts)
        previous = previous.index_select(2, flipped)
        starts_at_zero = starts == 0
        previous = torch.where(
            starts_at_zero[None, :, None], torch.zeros_like(previous), previous
        )
        if boundary_potential is None:
            boundary = unary.new_zeros((batch, end, 2))
        else:
            boundary = boundary_potential.index_select(1, starts)
        boundary = torch.where(
            starts_at_zero[None, :, None], start_potential[None, None, :], boundary
        )
        segment = (
            prefix[:, end, None, :] - prefix.index_select(1, starts)
            + run_potential[None, None, :]
            + _duration_values(duration_potential, durations, unary)[None, :, :]
            + boundary
        )
        end_valid = (lengths >= end)[:, None, None]
        candidates = torch.where(
            end_valid, previous + segment, torch.zeros_like(segment)
        )
        value = torch.logsumexp(candidates, dim=1)
        alpha.append(value.masked_fill(~end_valid[:, 0], -torch.inf))
    alpha_tensor = torch.stack(alpha, dim=1)
    log_partition = torch.logsumexp(
        alpha_tensor[torch.arange(batch, device=device), lengths], dim=-1
    )

    # beta[s,y] sums segmentations of [s,T) beginning in a y segment.
    beta: list[torch.Tensor | None] = [None] * (steps + 1)
    beta[steps] = unary.new_zeros((batch, 2))
    for start in range(steps - 1, -1, -1):
        ends = torch.arange(start + 1, steps + 1, device=device)
        durations = ends - start
        suffix_stack = torch.stack(beta[start + 1 :], dim=1)
        suffix = suffix_stack.index_select(2, flipped)
        ends_at_length = ends[None, :, None] == lengths[:, None, None]
        suffix = torch.where(ends_at_length, torch.zeros_like(suffix), suffix)
        suffix = suffix.masked_fill(
            ends[None, :, None] > lengths[:, None, None], -torch.inf
        )
        if boundary_potential is None or start == 0:
            boundary = unary.new_zeros((batch, 1, 2))
        else:
            boundary = boundary_potential[:, start : start + 1, :]
        if start == 0:
            boundary = start_potential[None, None, :].expand(batch, 1, 2)
        segment = (
            prefix.index_select(1, ends) - prefix[:, start, None, :]
            + run_potential[None, None, :]
            + _duration_values(duration_potential, durations, unary)[None, :, :]
            + boundary
        )
        start_valid = (lengths > start)[:, None, None]
        candidates = torch.where(
            start_valid, segment + suffix, torch.zeros_like(segment)
        )
        value = torch.logsumexp(candidates, dim=1)
        beta[start] = value.masked_fill(~start_valid[:, 0], -torch.inf)
    beta_tensor = torch.stack(beta, dim=1)

    # A segment posterior contributes to every covered block.  Keep these in
    # log space: the inherited native logits can exceed 50 in magnitude, so a
    # probability-space interval sum would underflow and clip confident scores.
    posterior_parts: list[torch.Tensor] = []
    start_parts: list[torch.Tensor] = []
    end_parts: list[torch.Tensor] = []
    for end in range(1, steps + 1):
        starts = torch.arange(end - 1, -1, -1, device=device)
        durations = end - starts
        previous = alpha_tensor.index_select(1, starts).index_select(2, flipped)
        starts_at_zero = starts == 0
        previous = torch.where(
            starts_at_zero[None, :, None], torch.zeros_like(previous), previous
        )
        if boundary_potential is None:
            boundary = unary.new_zeros((batch, end, 2))
        else:
            boundary = boundary_potential.index_select(1, starts)
        boundary = torch.where(
            starts_at_zero[None, :, None], start_potential[None, None, :], boundary
        )
        segment = (
            prefix[:, end, None, :] - prefix.index_select(1, starts)
            + run_potential[None, None, :]
            + _duration_values(duration_potential, durations, unary)[None, :, :]
            + boundary
        )
        suffix = beta_tensor[:, end : end + 1, :].index_select(2, flipped)
        suffix = suffix.expand(-1, end, -1)
        suffix = torch.where(
            (lengths == end)[:, None, None], torch.zeros_like(suffix), suffix
        )
        end_valid = (lengths >= end)[:, None, None]
        raw = previous + segment + suffix - log_partition[:, None, None]
        log_posterior = torch.where(
            end_valid, raw, torch.full_like(raw, -torch.inf)
        )
        posterior_parts.append(log_posterior)
        start_parts.append(starts)
        end_parts.append(torch.full_like(starts, end))

    log_posterior = torch.cat(posterior_parts, dim=1)
    starts = torch.cat(start_parts)
    ends = torch.cat(end_parts)
    log_marginal_parts = []
    for position in range(steps):
        covers = (starts <= position) & (ends > position)
        selected = log_posterior[:, covers, :]
        position_valid = valid[:, position, None, None]
        selected = torch.where(position_valid, selected, torch.zeros_like(selected))
        value = torch.logsumexp(selected, dim=1)
        log_marginal_parts.append(
            value.masked_fill(~position_valid[:, 0], 0.0)
        )
    log_marginals = torch.stack(log_marginal_parts, dim=1)
    return log_marginals, log_partition


def maximal_run_nll(
    unary: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    run_potential: torch.Tensor,
    start_potential: torch.Tensor,
    duration_potential: torch.Tensor | None = None,
    boundary_potential: torch.Tensor | None = None,
    log_partition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-valid-block NLL of the unique maximal-run gold segmentation."""
    batch, steps = _check_segmental_inputs(
        unary, lengths, run_potential, start_potential,
        duration_potential, boundary_potential,
    )
    if labels.shape != (batch, steps):
        raise ValueError(f"segmental labels must have shape {(batch, steps)}")
    if labels.device != unary.device:
        raise ValueError("segmental labels and potentials must share a device")
    positions = torch.arange(steps, device=unary.device)
    valid = positions[None, :] < lengths[:, None]
    if bool(((labels != 0) & (labels != 1) & valid).any()):
        raise ValueError("valid segmental labels must be binary")
    safe = labels.long().masked_fill(~valid, 0)
    if log_partition is None:
        _, log_partition = maximal_run_log_marginals(
            unary, lengths, run_potential, start_potential,
            duration_potential, boundary_potential,
        )
    if log_partition.shape != (batch,):
        raise ValueError("segmental log partition must have shape (B,)")

    unary_gold = unary.gather(2, safe.unsqueeze(-1)).squeeze(-1)
    gold_score = unary_gold.masked_fill(~valid, 0.0).sum(dim=1)
    run_start = valid.clone()
    if steps > 1:
        run_start[:, 1:] &= safe[:, 1:] != safe[:, :-1]
    run_score = run_potential[safe].masked_fill(~run_start, 0.0).sum(dim=1)
    gold_score = gold_score + run_score + start_potential[safe[:, 0]]

    if duration_potential is not None:
        remaining = torch.zeros(batch, dtype=torch.long, device=unary.device)
        run_lengths = torch.zeros_like(safe)
        for position in range(steps - 1, -1, -1):
            if position + 1 < steps:
                continues = valid[:, position + 1] & (
                    safe[:, position] == safe[:, position + 1]
                )
            else:
                continues = torch.zeros(batch, dtype=torch.bool, device=unary.device)
            remaining = torch.where(valid[:, position], torch.where(
                continues, remaining + 1, torch.ones_like(remaining)
            ), torch.zeros_like(remaining))
            run_lengths[:, position] = remaining
        buckets = torch.floor(torch.log2(run_lengths.clamp_min(1).float())).long()
        buckets = buckets.clamp_max(duration_potential.shape[1] - 1)
        duration_gold = duration_potential[safe, buckets]
        gold_score = gold_score + duration_gold.masked_fill(~run_start, 0.0).sum(dim=1)

    if boundary_potential is not None and steps > 1:
        boundary_gold = boundary_potential.gather(
            2, safe.unsqueeze(-1)
        ).squeeze(-1)
        transition_start = run_start.clone()
        transition_start[:, 0] = False
        gold_score = gold_score + boundary_gold.masked_fill(
            ~transition_start, 0.0
        ).sum(dim=1)
    return (log_partition - gold_score).sum() / lengths.float().sum()


class NativeSegmentalClassifier(NativeEventClassifier):
    """Frozen event parent plus exact cross-block maximal-run inference."""

    _DURATION_BINS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160,),
        *args,
        segmental_use_duration: bool = False,
        segmental_use_context_boundary: bool = False,
        segmental_boundary_hidden: int = 32,
        segmental_duration_bins: tuple[int, ...] = _DURATION_BINS,
        **kwargs,
    ):
        if type(segmental_use_duration) is not bool:
            raise ValueError("segmental duration switch must be boolean")
        if type(segmental_use_context_boundary) is not bool:
            raise ValueError("segmental context-boundary switch must be boolean")
        if segmental_use_context_boundary and not segmental_use_duration:
            raise ValueError("context-boundary arm must retain explicit duration")
        if segmental_boundary_hidden < 1:
            raise ValueError("segmental boundary width must be positive")
        if tuple(segmental_duration_bins) != self._DURATION_BINS:
            raise ValueError("segmental duration bins differ from fixed powers of two")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.segmental_use_duration = segmental_use_duration
        self.segmental_use_context_boundary = segmental_use_context_boundary
        self.segmental_boundary_hidden = int(segmental_boundary_hidden)
        self.segmental_duration_bins = tuple(segmental_duration_bins)
        self.segmental_start = nn.Parameter(torch.zeros(2))
        self.segmental_run = nn.Parameter(torch.zeros(2))
        if segmental_use_duration:
            # Bucket one is the fixed zero reference; run potentials absorb the
            # common intercept, leaving nonlinear duration identifiable.
            self.segmental_duration_residual = nn.Parameter(
                torch.zeros(2, len(self.segmental_duration_bins) - 1)
            )
        if segmental_use_context_boundary:
            self.segmental_boundary = nn.Sequential(
                nn.LayerNorm(3 * self.state_dim),
                nn.Linear(3 * self.state_dim, self.segmental_boundary_hidden),
                nn.GELU(),
                nn.Linear(self.segmental_boundary_hidden, 2),
            )
            nn.init.zeros_(self.segmental_boundary[-1].weight)
            nn.init.zeros_(self.segmental_boundary[-1].bias)
        self._segmental_aux: dict[str, torch.Tensor] = {}

    def _duration_potential(self, reference: torch.Tensor) -> torch.Tensor | None:
        if not self.segmental_use_duration:
            return None
        anchor = reference.new_zeros((2, 1))
        return torch.cat([anchor, self.segmental_duration_residual], dim=1)

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._segmental_aux = {}
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
        return self._apply_segmental_refinement(
            block_states, block_mask, parent_score
        )

    def _apply_segmental_refinement(
        self,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        parent_score: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the block-duration model to an already refined event score."""
        block_valid = block_mask.any(dim=-1)
        lengths = block_valid.sum(dim=-1).long()
        unary = torch.stack([-0.5 * parent_score, 0.5 * parent_score], dim=-1)
        boundary = None
        block_mean = (
            block_states * block_mask.unsqueeze(-1).to(block_states.dtype)
        ).sum(dim=2) / block_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        if self.segmental_use_context_boundary:
            pair = torch.cat([
                block_mean[:, :-1], block_mean[:, 1:],
                (block_mean[:, 1:] - block_mean[:, :-1]).abs(),
            ], dim=-1)
            transition = self.segmental_boundary(pair)
            boundary = torch.cat(
                [transition.new_zeros((transition.shape[0], 1, 2)), transition],
                dim=1,
            )
        duration = self._duration_potential(parent_score)
        log_marginals, log_partition = maximal_run_log_marginals(
            unary, lengths, self.segmental_run, self.segmental_start,
            duration, boundary,
        )
        structured_score = log_marginals[..., 1] - log_marginals[..., 0]
        self._segmental_aux.update(
            unary=unary, lengths=lengths, block_valid=block_valid,
            duration_potential=duration, boundary_potential=boundary,
            log_marginals=log_marginals, log_partition=log_partition,
            parent_score=parent_score, block_mean=block_mean,
        )
        return structured_score.masked_fill(~block_valid, 0.0)

    def segmental_nll(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """NLL for exact any-spoof labels on aligned eight-frame blocks."""
        if "block_valid" not in self._segmental_aux:
            raise RuntimeError("call the segmental classifier before segmental_nll")
        if (frame_labels.shape != valid.shape or valid.dtype != torch.bool
                or frame_labels.device != valid.device):
            raise ValueError("segmental frame labels and validity must align")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid segmental frame labels must be binary")
        safe = frame_labels.long().masked_fill(~valid, 1)
        blocked_labels = self._blockify(safe, 8, 1).amin(dim=-1)
        blocked_valid = self._blockify(valid, 8, False).any(dim=-1)
        if not torch.equal(blocked_valid, self._segmental_aux["block_valid"]):
            raise ValueError("segmental labels differ from the preceding forward")
        return maximal_run_nll(
            self._segmental_aux["unary"], blocked_labels,
            self._segmental_aux["lengths"], self.segmental_run,
            self.segmental_start,
            self._segmental_aux["duration_potential"],
            self._segmental_aux["boundary_potential"],
            self._segmental_aux["log_partition"],
        )


class NativeSegmentalLocalizer(NativeEventLocalizer):
    """Inherited fine reader with one segmental native-160 decision path."""

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
        )
        super().__init__(*args, **parent_args, **kwargs)
        inherited = self.native_pool.state_dict()
        pool_args = {
            key.removeprefix("native_"): value
            for key, value in parent_args.items()
        }
        self.native_pool = NativeSegmentalClassifier(
            self.out2.in_features,
            **pool_args,
            segmental_use_duration=native_segmental_use_duration,
            segmental_use_context_boundary=native_segmental_use_context_boundary,
            segmental_boundary_hidden=native_segmental_boundary_hidden,
            segmental_duration_bins=tuple(native_segmental_duration_bins),
        )
        self.native_pool.load_state_dict(inherited, strict=False)
