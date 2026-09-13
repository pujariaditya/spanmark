"""Boundary-event recurrence and interval geometry for native 160 ms scores.

The RED head predicts conditional spoof entry and exit probabilities.  Their
recurrence yields a coherent presence field and an exact probability that an
eight-frame block is entirely bonafide.  A zero-initialized residual keeps the
frozen event-pattern parent exactly unchanged before training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from spanmark.models.event import NativeEventClassifier, NativeEventLocalizer


def red_recurrence(
    initial_logits: torch.Tensor,
    transition_logits: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spoof presence, entry, exit, and log P(all bonafide).

    ``transition_logits[..., t, 0]`` is P(enter spoof | previously bonafide)
    and channel one is P(exit spoof | previously spoof).  Slot zero is
    represented by ``initial_logits``; transition slot zero is deliberately
    unused.  Valid slots must be a right-padded prefix.
    """
    if not initial_logits.is_floating_point():
        raise ValueError("RED initial logits must be floating point")
    if transition_logits.shape != valid.shape + (2,):
        raise ValueError("RED transition logits must match validity and have two channels")
    if initial_logits.shape != valid.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("RED initial logits and boolean validity do not align")
    if (initial_logits.device != transition_logits.device
            or initial_logits.device != valid.device):
        raise ValueError("RED tensors must share one device")
    if valid.shape[-1] != 8:
        raise ValueError("RED recurrence requires eight-frame native blocks")
    lengths = valid.sum(dim=-1)
    slots = torch.arange(8, device=valid.device)
    if not torch.equal(valid, slots < lengths.unsqueeze(-1)):
        raise ValueError("RED validity must be a right-padded prefix")

    dtype = initial_logits.dtype
    block_valid = valid[..., 0]
    initial = torch.sigmoid(initial_logits)
    previous = torch.where(block_valid, initial, torch.zeros_like(initial))
    presence = [previous]
    entry = [previous]
    exit_probability = [torch.zeros_like(previous)]
    log_all_bonafide = torch.where(
        block_valid, F.logsigmoid(-initial_logits), torch.zeros_like(initial_logits)
    )
    for slot in range(1, 8):
        slot_valid = valid[..., slot]
        enter = torch.sigmoid(transition_logits[..., slot, 0])
        leave = torch.sigmoid(transition_logits[..., slot, 1])
        entered = (1.0 - previous) * enter
        exited = previous * leave
        current = entered + previous * (1.0 - leave)
        presence.append(torch.where(slot_valid, current, torch.zeros_like(current)))
        entry.append(torch.where(slot_valid, entered, torch.zeros_like(entered)))
        exit_probability.append(torch.where(slot_valid, exited, torch.zeros_like(exited)))
        log_all_bonafide = log_all_bonafide + torch.where(
            slot_valid, F.logsigmoid(-transition_logits[..., slot, 0]),
            torch.zeros((), dtype=dtype, device=valid.device),
        )
        previous = torch.where(slot_valid, current, previous)
    return (
        torch.stack(presence, dim=-1),
        torch.stack(entry, dim=-1),
        torch.stack(exit_probability, dim=-1),
        log_all_bonafide,
    )


def interval_iou_from_extents(
    predicted: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """IoU of two inclusive intervals sharing their current-frame anchor."""
    if predicted.shape != target.shape or predicted.shape[-1] != 2:
        raise ValueError("interval extents must have identical (..., 2) shapes")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise ValueError("interval extents must be floating point")
    if predicted.device != target.device:
        raise ValueError("interval extents must share one device")
    if bool((predicted < 0).any()) or bool((target < 0).any()):
        raise ValueError("interval extents must be nonnegative")
    intersection = torch.minimum(predicted, target).sum(dim=-1) + 1.0
    union = torch.maximum(predicted, target).sum(dim=-1) + 1.0
    return intersection / union.clamp_min(torch.finfo(union.dtype).tiny)


class NativeREDClassifier(NativeEventClassifier):
    """Frozen-compatible event parent plus a recurrent boundary residual."""

    _RED_FEATURES = 9

    def __init__(
        self, state_dim: int, resolutions_ms: tuple[int, ...] = (160,), *args,
        red_hidden: int = 64, red_use_interval_proposal: bool = False,
        red_proposal_hidden: int = 32, red_proposal_layers: int = 2, **kwargs,
    ):
        if red_hidden < 1 or red_proposal_hidden < 1 or red_proposal_layers < 1:
            raise ValueError("RED widths and proposal depth must be positive")
        if type(red_use_interval_proposal) is not bool:
            raise ValueError("RED proposal switch must be boolean")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.red_hidden = int(red_hidden)
        self.red_use_interval_proposal = red_use_interval_proposal
        self.red_proposal_hidden = int(red_proposal_hidden)
        self.red_proposal_layers = int(red_proposal_layers)

        def event_head(output: int) -> nn.Sequential:
            head = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, self.red_hidden),
                nn.GELU(),
                nn.Linear(self.red_hidden, output),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        # These common modules are constructed in identical order in all arms.
        self.red_initial = event_head(1)
        self.red_transition = event_head(2)
        self.red_fusion = nn.Linear(self._RED_FEATURES, 1)
        nn.init.zeros_(self.red_fusion.weight)
        nn.init.zeros_(self.red_fusion.bias)
        if self.red_use_interval_proposal:
            self.red_proposal = nn.GRU(
                input_size=3,
                hidden_size=self.red_proposal_hidden,
                num_layers=self.red_proposal_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.red_proposal_out = nn.Linear(2 * self.red_proposal_hidden, 2)
        self._red_aux: dict[str, torch.Tensor] = {}

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._red_aux = {}
        result = super().forward(states, valid)
        self._red_aux["frame_valid"] = self._event_aux["frame_valid"]
        return result

    def _proposal_extents(
        self, presence: torch.Tensor, entry: torch.Tensor,
        exit_probability: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        shape = presence.shape
        if presence.numel() == 0:
            return presence.new_zeros(shape + (2,))
        features = torch.stack([presence, entry, exit_probability], dim=-1)
        flat = features.reshape(-1, 8, 3)
        lengths = valid.reshape(-1, 8).sum(dim=-1)
        nonempty = lengths > 0
        result = flat.new_zeros((flat.shape[0], 8, 2))
        if not bool(nonempty.any()):
            return result.reshape(shape + (2,))
        packed = pack_padded_sequence(
            flat[nonempty], lengths[nonempty].detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        packed_output, _ = self.red_proposal(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=8,
        )
        result[nonempty] = F.softplus(self.red_proposal_out(output))
        extents = result.reshape(shape + (2,))
        return extents.masked_fill(~valid.unsqueeze(-1), 0.0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        maximum = values.masked_fill(~mask.bool(), -torch.inf).amax(dim=-1)
        return maximum.masked_fill(~mask.bool().any(dim=-1), 0.0)

    def _refine_native_score(
        self, ms: int, block_states: torch.Tensor,
        block_mask: torch.Tensor, score: torch.Tensor,
    ) -> torch.Tensor:
        parent_score = super()._refine_native_score(ms, block_states, block_mask, score)
        initial_logits = self.red_initial(block_states[..., 0, :]).squeeze(-1)
        transition_logits = self.red_transition(block_states)
        presence, entry, exit_probability, log_all_bonafide = red_recurrence(
            initial_logits, transition_logits, block_mask,
        )
        block_valid = block_mask.any(dim=-1)
        epsilon = torch.finfo(log_all_bonafide.dtype).eps
        bounded_log_bonafide = log_all_bonafide.clamp_max(-epsilon)
        red_bonafide_logit = bounded_log_bonafide - torch.log(
            -torch.expm1(bounded_log_bonafide)
        )
        mask_float = block_mask.to(presence.dtype)
        transition_mask = block_mask.clone()
        transition_mask[..., 0] = False
        transition_float = transition_mask.to(presence.dtype)
        feature_parts = [
            red_bonafide_logit,
            self._masked_mean(presence, mask_float),
            self._masked_max(presence, block_mask),
            self._masked_mean(entry, mask_float),
            self._masked_max(entry, block_mask),
            self._masked_mean(exit_probability, transition_float),
            self._masked_max(exit_probability, transition_mask),
        ]
        proposal = None
        if self.red_use_interval_proposal:
            proposal = self._proposal_extents(
                presence, entry, exit_probability, block_mask,
            )
            span = (proposal.sum(dim=-1) + 1.0) / 8.0
            feature_parts.extend([
                self._masked_mean(span, mask_float),
                self._masked_max(span, block_mask),
            ])
        else:
            feature_parts.extend([torch.zeros_like(parent_score), torch.zeros_like(parent_score)])
        features = torch.stack(feature_parts, dim=-1)
        residual = self.red_fusion(features).squeeze(-1)
        self._red_aux.update(
            block_valid=block_mask, initial_logits=initial_logits,
            transition_logits=transition_logits,
            presence=presence, entry=entry, exit=exit_probability,
            log_all_bonafide=log_all_bonafide, red_features=features,
        )
        if proposal is not None:
            self._red_aux["proposal_extents"] = proposal
        return (parent_score + residual).masked_fill(~block_valid, 0.0)

    def _blocked_targets(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "frame_valid" not in self._red_aux:
            raise RuntimeError("call the RED classifier before requesting auxiliary losses")
        previous = self._red_aux["frame_valid"]
        if (frame_labels.shape != previous.shape or valid.shape != previous.shape
                or valid.dtype != torch.bool or frame_labels.device != previous.device
                or valid.device != previous.device or not torch.equal(valid, previous)):
            raise ValueError("RED labels and validity must match the preceding forward")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid RED labels must be 0=spoof or 1=bonafide")
        blocked_valid = self._blockify(valid, 8, False).bool()
        spoof = self._blockify((frame_labels == 0) & valid, 8, False).bool()
        return spoof, blocked_valid

    def presence_loss(self, frame_labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        spoof, blocked_valid = self._blocked_targets(frame_labels, valid)
        presence = self._red_aux["presence"]
        if not bool(blocked_valid.any()):
            return presence.sum() * 0.0
        return F.binary_cross_entropy(
            presence[blocked_valid], spoof.to(presence.dtype)[blocked_valid]
        )

    def onset_offset_loss(
        self, frame_labels: torch.Tensor, valid: torch.Tensor, *, focal_gamma: float = 2.0,
    ) -> torch.Tensor:
        if not isinstance(focal_gamma, (int, float)) or focal_gamma < 0:
            raise ValueError("RED focal gamma must be nonnegative")
        spoof, blocked_valid = self._blocked_targets(frame_labels, valid)
        transition_valid = blocked_valid.clone()
        transition_valid[..., 0] = False
        if not bool(transition_valid.any()):
            return self._red_aux["transition_logits"].sum() * 0.0
        onset = spoof & ~torch.roll(spoof, shifts=1, dims=-1)
        offset = ~spoof & torch.roll(spoof, shifts=1, dims=-1)
        target = torch.stack([onset, offset], dim=-1).to(
            self._red_aux["transition_logits"].dtype
        )
        logits = self._red_aux["transition_logits"]
        per_channel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_true = torch.exp(-per_channel)
        focal = (1.0 - p_true).pow(float(focal_gamma)) * per_channel
        return focal[transition_valid].mean()

    def interval_iou_loss(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
        target_left: torch.Tensor, target_right: torch.Tensor,
        inverse_run_length: torch.Tensor,
    ) -> torch.Tensor:
        if "proposal_extents" not in self._red_aux:
            raise RuntimeError("this RED arm has no interval proposal")
        spoof, blocked_valid = self._blocked_targets(frame_labels, valid)
        for name, value in (
            ("target_left", target_left), ("target_right", target_right),
            ("inverse_run_length", inverse_run_length),
        ):
            if (value.shape != valid.shape or not value.is_floating_point()
                    or value.device != valid.device):
                raise ValueError(f"{name} must be a matching floating tensor")
            if bool((value < 0).any()) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite and nonnegative")
        left = self._blockify(target_left, 8, 0.0)
        right = self._blockify(target_right, 8, 0.0)
        weights = self._blockify(inverse_run_length, 8, 0.0)
        supervised = spoof & blocked_valid & (weights > 0)
        proposal = self._red_aux["proposal_extents"]
        if not bool(supervised.any()):
            return proposal.sum() * 0.0
        target = torch.stack([left, right], dim=-1)
        iou = interval_iou_from_extents(proposal, target)
        weighted = (1.0 - iou) * weights
        return weighted[supervised].sum() / weights[supervised].sum().clamp_min(
            torch.finfo(weights.dtype).tiny
        )


class NativeREDLocalizer(NativeEventLocalizer):
    """The inherited XLS-R reader with one RED-refined native 160 ms path."""

    def __init__(
        self, *args, native_resolutions_ms: tuple[int, ...] = (160,),
        native_attention_hidden: int = 64, native_classifier_hidden: int = 128,
        native_dropout: float = 0.1, native_trunk_layers: int = 0,
        native_trunk_bottleneck: int = 64, native_zero_output_init: bool = True,
        native_summary_mode: str = "meanstd", native_block_layers: int = 0,
        native_block_bottleneck: int = 64, native_event_hidden: int = 64,
        native_event_pairwise: bool = True, native_red_hidden: int = 64,
        native_red_use_interval_proposal: bool = False,
        native_red_proposal_hidden: int = 32, native_red_proposal_layers: int = 2,
        **kwargs,
    ):
        pool_args = dict(
            resolutions_ms=tuple(native_resolutions_ms),
            attention_hidden=native_attention_hidden,
            classifier_hidden=native_classifier_hidden,
            dropout=native_dropout,
            trunk_layers=native_trunk_layers,
            trunk_bottleneck=native_trunk_bottleneck,
            zero_output_init=native_zero_output_init,
            summary_mode=native_summary_mode,
            block_layers=native_block_layers,
            block_bottleneck=native_block_bottleneck,
            event_hidden=native_event_hidden,
            event_pairwise=native_event_pairwise,
        )
        super().__init__(
            *args,
            native_resolutions_ms=native_resolutions_ms,
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
            **kwargs,
        )
        inherited = self.native_pool.state_dict()
        self.native_pool = NativeREDClassifier(
            self.out2.in_features, **pool_args,
            red_hidden=native_red_hidden,
            red_use_interval_proposal=native_red_use_interval_proposal,
            red_proposal_hidden=native_red_proposal_hidden,
            red_proposal_layers=native_red_proposal_layers,
        )
        missing, unexpected = self.native_pool.load_state_dict(inherited, strict=False)
        if unexpected or not missing or any(not key.startswith("red_") for key in missing):
            raise RuntimeError(f"RED constructor changed its event parent: {missing}/{unexpected}")
