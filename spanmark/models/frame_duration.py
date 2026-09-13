"""Exact 20 ms run-duration evidence for the native 160 ms decision.

The frozen event parent supplies one spoof energy per frame.  A deterministic
label/run-age automaton couples those energies across the whole utterance and
computes the exact partition of two events for every aligned block: all frames
bonafide versus at least one spoof frame.  Every binary label sequence has one
automaton path.  Zero duration potentials therefore recover the parent's local
event partition exactly; learned duration evidence enters before the inherited
block-level semi-Markov refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spanmark.models.event import NativeEventClassifier
from spanmark.models.segmental import NativeSegmentalClassifier, NativeSegmentalLocalizer


def _check_inputs(
    spoof_unary: torch.Tensor,
    lengths: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
) -> tuple[int, int, int]:
    if spoof_unary.ndim != 2 or not spoof_unary.is_floating_point():
        raise ValueError("spoof unary must have shape (B,T)")
    batch, steps = spoof_unary.shape
    if steps < 1 or tuple(lengths.shape) != (batch,):
        raise ValueError("frame lengths must have shape (B,) and T must be positive")
    if lengths.device != spoof_unary.device:
        raise ValueError("frame lengths and unaries must share a device")
    if bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError("every frame length must lie in [1,T]")
    if (duration_end.ndim != 2 or duration_end.shape[0] != 2
            or duration_end.shape[1] < 1 or not duration_end.is_floating_point()):
        raise ValueError("duration end potentials must have shape (2,K), K>=1")
    if tuple(tail_slope.shape) != (2,) or not tail_slope.is_floating_point():
        raise ValueError("tail slopes must have shape (2,)")
    if duration_end.device != spoof_unary.device or tail_slope.device != spoof_unary.device:
        raise ValueError("all frame-duration tensors must share a device")
    return batch, steps, duration_end.shape[1]


def _emissions(spoof_unary: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(spoof_unary.shape[1], device=spoof_unary.device)
    valid = positions[None, :] < lengths[:, None]
    clean = spoof_unary.masked_fill(~valid, 0.0)
    return torch.stack([torch.zeros_like(clean), clean], dim=-1)


def _initial(emission: torch.Tensor, horizon: int) -> torch.Tensor:
    """Only age one is legal at the first frame."""
    padding = emission.new_full((*emission.shape, horizon - 1), -torch.inf)
    return torch.cat([emission.unsqueeze(-1), padding], dim=-1)


def _safe_logaddexp(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Log-add without NaN gradients when both structural paths are absent."""
    absent = torch.isneginf(left) & torch.isneginf(right)
    safe_left = left.masked_fill(absent, 0.0)
    safe_right = right.masked_fill(absent, 0.0)
    return torch.logaddexp(safe_left, safe_right).masked_fill(absent, -torch.inf)


def _safe_logsumexp(
    values: torch.Tensor, dim: int | tuple[int, ...]
) -> torch.Tensor:
    """Reduce log weights without NaN gradients for an unreachable group."""
    dimensions = (dim,) if isinstance(dim, int) else dim
    dimensions = tuple(axis % values.ndim for axis in dimensions)
    absent = torch.isneginf(values).all(dim=dimensions, keepdim=True)
    safe_values = values.masked_fill(absent, 0.0)
    result = torch.logsumexp(safe_values, dim=dimensions)
    reduced_absent = absent
    for axis in sorted(dimensions, reverse=True):
        reduced_absent = reduced_absent.squeeze(axis)
    return result.masked_fill(reduced_absent, -torch.inf)


def _advance(
    alpha: torch.Tensor,
    emission: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
) -> torch.Tensor:
    """One sparse automaton step for tensors ending in ``(...,2,K)``."""
    horizon = duration_end.shape[1]
    rows = []
    for label in (0, 1):
        other = 1 - label
        switched = _safe_logsumexp(
            alpha[..., other, :] + duration_end[other], dim=-1
        )
        if horizon == 1:
            continued = alpha[..., label, 0] + tail_slope[label]
            values = _safe_logaddexp(switched, continued).unsqueeze(-1)
        else:
            age_one = switched.unsqueeze(-1)
            middle = alpha[..., label, : horizon - 2]
            saturated = _safe_logaddexp(
                alpha[..., label, horizon - 2],
                alpha[..., label, horizon - 1] + tail_slope[label],
            ).unsqueeze(-1)
            values = torch.cat([age_one, middle, saturated], dim=-1)
        rows.append(values + emission[..., label, None])
    return torch.stack(rows, dim=-2)


def _retreat(
    beta_next: torch.Tensor,
    next_emission: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
) -> torch.Tensor:
    """Backward message from one current run-age state."""
    horizon = duration_end.shape[1]
    rows = []
    for label in (0, 1):
        other = 1 - label
        next_ages = torch.arange(horizon, device=beta_next.device) + 1
        next_ages = next_ages.clamp_max(horizon - 1)
        continued = beta_next[..., label, :].index_select(-1, next_ages)
        continued = continued + next_emission[..., label, None]
        tail = torch.zeros(horizon, dtype=continued.dtype, device=continued.device)
        tail[-1] = tail_slope[label]
        continued = continued + tail
        switched = (
            duration_end[label]
            + next_emission[..., other, None]
            + beta_next[..., other, :1]
        )
        rows.append(_safe_logaddexp(continued, switched))
    return torch.stack(rows, dim=-2)


def run_age_messages(
    spoof_unary: torch.Tensor,
    lengths: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return emissions, inclusive forward/backward messages, and log Z."""
    batch, steps, horizon = _check_inputs(
        spoof_unary, lengths, duration_end, tail_slope
    )
    emissions = _emissions(spoof_unary, lengths)
    current = _initial(emissions[:, 0], horizon)
    forward = [current]
    for position in range(1, steps):
        proposal = _advance(
            current, emissions[:, position], duration_end, tail_slope
        )
        active = (lengths > position)[:, None, None]
        current = torch.where(active, proposal, current)
        forward.append(current)
    forward_messages = torch.stack(forward, dim=1)
    log_partition = _safe_logsumexp(
        current + duration_end[None, :, :], dim=(-2, -1)
    )

    terminal = duration_end[None, :, :].expand(batch, -1, -1)
    backward: list[torch.Tensor | None] = [None] * steps
    current = terminal
    for position in range(steps - 1, -1, -1):
        if position == steps - 1:
            proposal = terminal
        else:
            proposal = _retreat(
                current, emissions[:, position + 1], duration_end, tail_slope
            )
        is_terminal = (lengths == position + 1)[:, None, None]
        has_frame = (lengths > position)[:, None, None]
        current = torch.where(is_terminal, terminal, proposal)
        current = torch.where(has_frame, current, terminal)
        backward[position] = current
    backward_messages = torch.stack(backward, dim=1)
    return emissions, forward_messages, backward_messages, log_partition


def aligned_event_log_partitions(
    spoof_unary: torch.Tensor,
    lengths: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
    *,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact log Z for all-bonafide and any-spoof events in aligned blocks."""
    if block_size < 1:
        raise ValueError("block size must be positive")
    emissions, forward, backward, log_partition = run_age_messages(
        spoof_unary, lengths, duration_end, tail_slope
    )
    batch, steps, _ = emissions.shape
    horizon = duration_end.shape[1]
    negative = spoof_unary.new_full((batch, 2, horizon), -torch.inf)
    all_parts = []
    any_parts = []
    batch_index = torch.arange(batch, device=spoof_unary.device)
    for start in range(0, steps, block_size):
        if start == 0:
            current = _initial(emissions[:, 0], horizon)
        else:
            current = _advance(
                forward[:, start - 1], emissions[:, start],
                duration_end, tail_slope,
            )
        no_spoof = torch.where(
            torch.tensor([True, False], device=spoof_unary.device)[None, :, None],
            current, negative,
        )
        seen_spoof = torch.where(
            torch.tensor([False, True], device=spoof_unary.device)[None, :, None],
            current, negative,
        )
        stop = min(start + block_size, steps)
        for position in range(start + 1, stop):
            no_proposal = _advance(
                no_spoof, emissions[:, position], duration_end, tail_slope
            )
            seen_proposal = _advance(
                seen_spoof, emissions[:, position], duration_end, tail_slope
            )
            from_no_to_spoof = torch.where(
                torch.tensor([False, True], device=spoof_unary.device)[None, :, None],
                no_proposal, negative,
            )
            next_no = torch.where(
                torch.tensor([True, False], device=spoof_unary.device)[None, :, None],
                no_proposal, negative,
            )
            next_seen = _safe_logaddexp(seen_proposal, from_no_to_spoof)
            active = (lengths > position)[:, None, None]
            no_spoof = torch.where(active, next_no, no_spoof)
            seen_spoof = torch.where(active, next_seen, seen_spoof)
        end_index = torch.minimum(
            lengths, lengths.new_full(lengths.shape, stop)
        ) - 1
        suffix = backward[batch_index, end_index]
        active_block = lengths > start
        log_all = _safe_logsumexp(no_spoof + suffix, dim=(-2, -1))
        log_any = _safe_logsumexp(seen_spoof + suffix, dim=(-2, -1))
        all_parts.append(log_all.masked_fill(~active_block, 0.0))
        any_parts.append(log_any.masked_fill(~active_block, 0.0))
    return torch.stack(all_parts, dim=1), torch.stack(any_parts, dim=1), log_partition


def run_age_nll(
    spoof_unary: torch.Tensor,
    spoof_labels: torch.Tensor,
    lengths: torch.Tensor,
    duration_end: torch.Tensor,
    tail_slope: torch.Tensor,
    log_partition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-valid-frame NLL of the unique gold run-age path."""
    _, steps, horizon = _check_inputs(
        spoof_unary, lengths, duration_end, tail_slope
    )
    if spoof_labels.shape != spoof_unary.shape or spoof_labels.device != spoof_unary.device:
        raise ValueError("spoof labels must match the unary tensor")
    positions = torch.arange(steps, device=spoof_unary.device)
    valid = positions[None, :] < lengths[:, None]
    if bool(((spoof_labels != 0) & (spoof_labels != 1) & valid).any()):
        raise ValueError("valid spoof labels must be binary")
    labels = spoof_labels.long().masked_fill(~valid, 0)
    if log_partition is None:
        *_, log_partition = run_age_messages(
            spoof_unary, lengths, duration_end, tail_slope
        )
    gold = spoof_unary.masked_fill(~valid | (labels == 0), 0.0).sum(dim=1)
    age = torch.ones(labels.shape[0], dtype=torch.long, device=labels.device)
    for position in range(steps):
        if position:
            same = valid[:, position] & valid[:, position - 1] & (
                labels[:, position] == labels[:, position - 1]
            )
            tail = same & (age == horizon)
            gold = gold + torch.where(
                tail, tail_slope[labels[:, position]], torch.zeros_like(gold)
            )
            age = torch.where(
                same, (age + 1).clamp_max(horizon), torch.ones_like(age)
            )
        if position + 1 == steps:
            run_end = valid[:, position]
        else:
            run_end = valid[:, position] & (
                ~valid[:, position + 1]
                | (labels[:, position] != labels[:, position + 1])
            )
        ended = duration_end[labels[:, position], age - 1]
        gold = gold + torch.where(run_end, ended, torch.zeros_like(gold))
    return (log_partition - gold).sum() / lengths.float().sum()


class NativeFrameDurationClassifier(NativeSegmentalClassifier):
    """Frozen block-duration parent plus pre-collapse frame-run duration."""

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160,),
        *args,
        frame_duration_horizon: int = 8,
        **kwargs,
    ):
        if not isinstance(frame_duration_horizon, int) or frame_duration_horizon < 1:
            raise ValueError("frame duration horizon must be a positive integer")
        if kwargs.get("segmental_use_duration") is not True:
            raise ValueError("frame duration requires the explicit block-duration parent")
        if kwargs.get("segmental_use_context_boundary") is not False:
            raise ValueError("frame duration requires the context-free block parent")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.frame_duration_horizon = frame_duration_horizon
        self.frame_duration_end = nn.Parameter(torch.zeros(2, frame_duration_horizon))
        self.frame_duration_tail = nn.Parameter(torch.zeros(2))
        self._frame_duration_aux: dict[str, torch.Tensor] = {}

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._frame_duration_aux = {}
        return super().forward(states, valid)

    def _refine_native_score(
        self,
        ms: int,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        local_event_score = NativeEventClassifier._refine_native_score(
            self, ms, block_states, block_mask, score
        )
        frame_logits = self._event_aux["frame_logits"].reshape(
            block_states.shape[0], -1
        )
        lengths = block_mask.reshape(block_mask.shape[0], -1).sum(dim=-1).long()
        log_all, log_any, log_partition = aligned_event_log_partitions(
            frame_logits, lengths, self.frame_duration_end,
            self.frame_duration_tail, block_size=8,
        )
        local_log_any = self._event_aux["logZ_any"]
        global_event_log_odds = log_any - log_all
        event_delta = global_event_log_odds - local_log_any
        global_event_score = local_event_score - event_delta
        self._frame_duration_aux.update(
            frame_logits=frame_logits, lengths=lengths,
            log_all=log_all, log_any=log_any,
            log_partition=log_partition, local_log_any=local_log_any,
            global_event_log_odds=global_event_log_odds,
            event_delta=event_delta, local_event_score=local_event_score,
            global_event_score=global_event_score,
        )
        return self._apply_segmental_refinement(
            block_states, block_mask, global_event_score
        )

    def frame_duration_nll(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        if "frame_logits" not in self._frame_duration_aux:
            raise RuntimeError("call the frame-duration classifier before its NLL")
        if (frame_labels.shape != valid.shape or valid.dtype != torch.bool
                or frame_labels.device != valid.device):
            raise ValueError("frame-duration labels and validity must align")
        expected_lengths = valid.sum(dim=-1).long()
        if not torch.equal(expected_lengths, self._frame_duration_aux["lengths"]):
            raise ValueError("frame-duration validity differs from the preceding forward")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid frame labels must use 0=spoof and 1=bonafide")
        logits = self._frame_duration_aux["frame_logits"][:, : valid.shape[1]]
        spoof = (frame_labels == 0).long().masked_fill(~valid, 0)
        return run_age_nll(
            logits, spoof, expected_lengths, self.frame_duration_end,
            self.frame_duration_tail,
            self._frame_duration_aux["log_partition"],
        )


class NativeFrameDurationLocalizer(NativeSegmentalLocalizer):
    """Inherited XLS-R reader with one frame-duration native160 model."""

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
        native_frame_duration_horizon: int = 8,
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
        self.native_pool = NativeFrameDurationClassifier(
            self.out2.in_features, **pool_args,
            frame_duration_horizon=native_frame_duration_horizon,
        )
        self.native_pool.load_state_dict(inherited, strict=False)
