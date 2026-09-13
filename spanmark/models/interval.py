"""Dense anchor-free interval geometry for native 160 ms localization.

Every valid 20 ms point predicts spoof confidence, centerness, and distances
to the enclosing spoof run.  Decoded candidates are compared with every
aligned 160 ms block inside the model; a zero-start residual preserves the
frozen block-duration parent exactly before training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from spanmark.models.segmental import NativeSegmentalClassifier, NativeSegmentalLocalizer


def interval_diou_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return 1D distance-IoU loss for intervals sharing their anchor point.

    Extents are nonnegative distances from the anchor to inclusive left/right
    endpoints.  The center-distance term keeps a gradient when overlap is poor
    or one predicted extent approaches zero.
    """
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
    iou = intersection / union.clamp_min(torch.finfo(union.dtype).tiny)
    predicted_center = 0.5 * (predicted[..., 1] - predicted[..., 0])
    target_center = 0.5 * (target[..., 1] - target[..., 0])
    enclosing = union
    center_penalty = (predicted_center - target_center).square() / enclosing.square().clamp_min(
        torch.finfo(union.dtype).tiny
    )
    return 1.0 - iou + center_penalty


def sample_temporal_features(
    features: torch.Tensor,
    positions: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear feature samples at continuous temporal positions."""
    if features.ndim != 3 or positions.shape != features.shape[:2]:
        raise ValueError("temporal samples require features (B,T,D) and positions (B,T)")
    if valid.shape != positions.shape or valid.dtype != torch.bool:
        raise ValueError("temporal sample validity must match positions")
    if features.device != positions.device or features.device != valid.device:
        raise ValueError("temporal sample tensors must share one device")
    if not positions.is_floating_point():
        raise ValueError("temporal sample positions must be floating point")
    batch, steps, width = features.shape
    if steps == 0:
        return features.new_zeros((batch, 0, width))
    lengths = valid.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("temporal sampling requires nonempty right-padded sequences")
    slots = torch.arange(steps, device=valid.device)
    if not torch.equal(valid, slots < lengths.unsqueeze(-1)):
        raise ValueError("temporal sample validity must be a right-padded prefix")
    maximum = (lengths - 1).to(positions.dtype).unsqueeze(-1)
    clipped = torch.minimum(positions.clamp_min(0.0), maximum)
    lower = clipped.floor().long()
    upper = torch.minimum(lower + 1, (lengths - 1).unsqueeze(-1))
    gather_shape = (-1, -1, width)
    lower_value = features.gather(1, lower.unsqueeze(-1).expand(*gather_shape))
    upper_value = features.gather(1, upper.unsqueeze(-1).expand(*gather_shape))
    weight = (clipped - lower.to(clipped.dtype)).unsqueeze(-1)
    sampled = lower_value + weight * (upper_value - lower_value)
    return sampled.masked_fill(~valid.unsqueeze(-1), 0.0)


def candidate_block_probability(
    spoof_logits: torch.Tensor,
    centerness_logits: torch.Tensor,
    extents: torch.Tensor,
    valid: torch.Tensor,
    *,
    block_size: int = 8,
    intersection_temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximum confidence-weighted soft interval intersection per block."""
    if spoof_logits.shape != centerness_logits.shape or valid.shape != spoof_logits.shape:
        raise ValueError("candidate logits and validity must have identical (B,T) shape")
    if extents.shape != spoof_logits.shape + (2,):
        raise ValueError("candidate extents must have shape (B,T,2)")
    if valid.dtype != torch.bool or block_size < 1 or intersection_temperature <= 0:
        raise ValueError("candidate block configuration is invalid")
    if not spoof_logits.is_floating_point() or not centerness_logits.is_floating_point():
        raise ValueError("candidate logits must be floating point")
    if (spoof_logits.device != centerness_logits.device
            or spoof_logits.device != extents.device
            or spoof_logits.device != valid.device):
        raise ValueError("candidate tensors must share one device")
    batch, steps = spoof_logits.shape
    blocks = (steps + block_size - 1) // block_size
    block_valid = F.pad(valid, (0, (-steps) % block_size), value=False).view(
        batch, blocks, block_size
    ).any(dim=-1)
    if steps == 0:
        return spoof_logits.new_zeros((batch, 0)), block_valid
    if bool((extents < 0).any()):
        raise ValueError("candidate extents must be nonnegative")
    lengths = valid.sum(dim=-1)
    slots = torch.arange(steps, device=valid.device)
    if not torch.equal(valid, slots < lengths.unsqueeze(-1)):
        raise ValueError("candidate validity must be a right-padded prefix")

    dtype = spoof_logits.dtype
    candidate_position = slots.to(dtype)[None, :, None]
    candidate_start = candidate_position - extents[..., 0, None]
    candidate_end = candidate_position + extents[..., 1, None]
    block_start = (torch.arange(blocks, device=valid.device) * block_size).to(dtype)
    block_length = F.pad(
        valid, (0, (-steps) % block_size), value=False
    ).view(batch, blocks, block_size).sum(dim=-1)
    block_end = block_start[None, :] + block_length.to(dtype) - 1.0
    temperature = float(intersection_temperature)
    left_gate = torch.sigmoid((candidate_end - block_start[None, None, :] + 0.5) / temperature)
    right_gate = torch.sigmoid((block_end[:, None, :] - candidate_start + 0.5) / temperature)
    confidence = (
        torch.sigmoid(spoof_logits) * torch.sigmoid(centerness_logits)
    )[..., None]
    candidate = confidence * left_gate * right_gate
    candidate = candidate.masked_fill(~valid[..., None], 0.0)
    candidate = candidate.masked_fill(~block_valid[:, None, :], 0.0)
    probability = candidate.amax(dim=1)
    return probability.masked_fill(~block_valid, 0.0), block_valid


class NativeIntervalClassifier(NativeSegmentalClassifier):
    """Frozen duration parent plus a dense point-to-interval residual."""

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160,),
        *args,
        interval_hidden: int = 64,
        interval_use_context: bool = False,
        interval_context_hidden: int = 32,
        interval_context_layers: int = 2,
        interval_use_boundary_refine: bool = False,
        interval_extent_scale: float = 32.0,
        interval_intersection_temperature: float = 0.5,
        **kwargs,
    ):
        if tuple(resolutions_ms) != (160,):
            raise ValueError("the interval classifier supplies native 160 ms only")
        if interval_hidden < 1 or interval_context_hidden < 1 or interval_context_layers < 1:
            raise ValueError("interval widths and context depth must be positive")
        if type(interval_use_context) is not bool or type(interval_use_boundary_refine) is not bool:
            raise ValueError("interval arm switches must be boolean")
        if interval_use_boundary_refine and not interval_use_context:
            raise ValueError("boundary refinement requires whole-utterance context")
        if interval_extent_scale <= 0 or interval_intersection_temperature <= 0:
            raise ValueError("interval scales must be positive")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.interval_hidden = int(interval_hidden)
        self.interval_use_context = interval_use_context
        self.interval_context_hidden = int(interval_context_hidden)
        self.interval_context_layers = int(interval_context_layers)
        self.interval_use_boundary_refine = interval_use_boundary_refine
        self.interval_extent_scale = float(interval_extent_scale)
        self.interval_intersection_temperature = float(interval_intersection_temperature)

        # Common modules are constructed before optional arms so their initial
        # tensors are bit-identical under the shared constructor seed.
        self.interval_projection = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.interval_hidden),
            nn.GELU(),
        )
        self.interval_spoof = nn.Linear(self.interval_hidden, 1)
        self.interval_centerness = nn.Linear(self.interval_hidden, 1)
        self.interval_extent = nn.Linear(self.interval_hidden, 2)
        self.interval_fusion = nn.Linear(1, 1)
        nn.init.zeros_(self.interval_fusion.weight)
        nn.init.zeros_(self.interval_fusion.bias)
        if self.interval_use_context:
            if 2 * self.interval_context_hidden != self.interval_hidden:
                raise ValueError("bidirectional context width must equal interval hidden width")
            self.interval_context = nn.GRU(
                input_size=self.interval_hidden,
                hidden_size=self.interval_context_hidden,
                num_layers=self.interval_context_layers,
                batch_first=True,
                bidirectional=True,
            )
        if self.interval_use_boundary_refine:
            self.interval_boundary = nn.Sequential(
                nn.LayerNorm(3 * self.interval_hidden),
                nn.Linear(3 * self.interval_hidden, self.interval_hidden),
                nn.GELU(),
                nn.Linear(self.interval_hidden, 4),
            )
            nn.init.zeros_(self.interval_boundary[-1].weight)
            nn.init.zeros_(self.interval_boundary[-1].bias)
        self._interval_aux: dict[str, torch.Tensor] = {}

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._interval_aux = {}
        result = super().forward(states, valid)
        # The native parent pads to a multiple of eight internally.  Auxiliary
        # labels retain the caller's original length, so keep both masks.
        self._interval_aux["frame_valid"] = valid
        return result

    def _contextualize(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not self.interval_use_context:
            return hidden
        if hidden.numel() == 0:
            return hidden
        lengths = valid.sum(dim=-1)
        nonempty = lengths > 0
        output = hidden.new_zeros(hidden.shape)
        if bool(nonempty.any()):
            packed = pack_padded_sequence(
                hidden[nonempty], lengths[nonempty].detach().cpu(),
                batch_first=True, enforce_sorted=False,
            )
            packed_output, _ = self.interval_context(packed)
            unpacked, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=hidden.shape[1],
            )
            output[nonempty] = unpacked
        return output.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _apply_interval_refinement(
        self,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        parent_score: torch.Tensor,
    ) -> torch.Tensor:
        batch, blocks, frames, width = block_states.shape
        sequence = block_states.reshape(batch, blocks * frames, width)
        valid = block_mask.reshape(batch, blocks * frames)
        sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0.0)
        hidden = self._contextualize(self.interval_projection(sequence), valid)
        spoof_logits = self.interval_spoof(hidden).squeeze(-1)
        centerness_logits = self.interval_centerness(hidden).squeeze(-1)
        extent_raw = self.interval_extent(hidden)
        extents = F.softplus(extent_raw) * self.interval_extent_scale

        boundary_features = None
        if self.interval_use_boundary_refine:
            positions = torch.arange(sequence.shape[1], device=sequence.device).to(
                extents.dtype
            )[None, :]
            start = positions - extents[..., 0]
            end = positions + extents[..., 1]
            start_features = sample_temporal_features(hidden, start, valid)
            end_features = sample_temporal_features(hidden, end, valid)
            boundary_features = torch.cat([hidden, start_features, end_features], dim=-1)
            delta = self.interval_boundary(boundary_features)
            spoof_logits = spoof_logits + delta[..., 0]
            centerness_logits = centerness_logits + delta[..., 1]
            extent_raw = extent_raw + delta[..., 2:]
            extents = F.softplus(extent_raw) * self.interval_extent_scale

        probability, candidate_valid = candidate_block_probability(
            spoof_logits, centerness_logits, extents, valid,
            block_size=frames,
            intersection_temperature=self.interval_intersection_temperature,
        )
        epsilon = torch.finfo(probability.dtype).eps
        bounded = probability.clamp(min=epsilon, max=1.0 - epsilon)
        bonafide_logit = torch.log1p(-bounded) - torch.log(bounded)
        residual = self.interval_fusion(bonafide_logit.unsqueeze(-1)).squeeze(-1)
        self._interval_aux.update(
            sequence_valid=valid,
            hidden=hidden,
            spoof_logits=spoof_logits,
            centerness_logits=centerness_logits,
            extents=extents,
            candidate_probability=probability,
            candidate_bonafide_logit=bonafide_logit,
            block_valid=candidate_valid,
        )
        if boundary_features is not None:
            self._interval_aux["boundary_features"] = boundary_features
        return (parent_score + residual).masked_fill(~candidate_valid, 0.0)

    def _refine_native_score(
        self,
        ms: int,
        block_states: torch.Tensor,
        block_mask: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        parent_score = super()._refine_native_score(ms, block_states, block_mask, score)
        return self._apply_interval_refinement(block_states, block_mask, parent_score)

    def interval_losses(
        self,
        frame_labels: torch.Tensor,
        valid: torch.Tensor,
        target_left: torch.Tensor,
        target_right: torch.Tensor,
        *,
        focal_gamma: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spoof focal, spoof centerness, and extent DIoU losses."""
        if "frame_valid" not in self._interval_aux:
            raise RuntimeError("call the interval classifier before interval_losses")
        previous = self._interval_aux["frame_valid"]
        for name, value in (
            ("frame_labels", frame_labels), ("valid", valid),
            ("target_left", target_left), ("target_right", target_right),
        ):
            if value.shape != previous.shape or value.device != previous.device:
                raise ValueError(f"{name} must match the preceding interval forward")
        if valid.dtype != torch.bool or not torch.equal(valid, previous):
            raise ValueError("interval loss validity must match the preceding forward")
        if not target_left.is_floating_point() or not target_right.is_floating_point():
            raise ValueError("interval extent targets must be floating point")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid interval labels must be 0=spoof or 1=bonafide")
        if not isinstance(focal_gamma, (int, float)) or focal_gamma < 0:
            raise ValueError("interval focal gamma must be nonnegative")
        if not bool(valid.any()):
            zero = self._interval_aux["spoof_logits"].sum() * 0.0
            return zero, zero, zero

        spoof = (frame_labels == 0) & valid
        steps = valid.shape[1]
        spoof_logits = self._interval_aux["spoof_logits"][:, :steps]
        target = spoof.to(spoof_logits.dtype)
        per_frame = F.binary_cross_entropy_with_logits(
            spoof_logits, target, reduction="none"
        )
        probability_true = torch.exp(-per_frame)
        focal = ((1.0 - probability_true).pow(float(focal_gamma)) * per_frame)[valid].mean()
        if not bool(spoof.any()):
            zero = self._interval_aux["centerness_logits"][:, :steps].sum() * 0.0
            return focal, zero, self._interval_aux["extents"][:, :steps].sum() * 0.0

        target_extents = torch.stack([target_left, target_right], dim=-1).to(
            self._interval_aux["extents"].dtype
        )
        shorter = torch.minimum(target_left, target_right) + 1.0
        longer = torch.maximum(target_left, target_right) + 1.0
        centerness_target = torch.sqrt(shorter / longer.clamp_min(1.0)).to(
            self._interval_aux["centerness_logits"].dtype
        )
        centerness = F.binary_cross_entropy_with_logits(
            self._interval_aux["centerness_logits"][:, :steps][spoof],
            centerness_target[spoof],
        )
        diou = interval_diou_loss(
            self._interval_aux["extents"][:, :steps][spoof], target_extents[spoof]
        ).mean()
        return focal, centerness, diou


class NativeIntervalLocalizer(NativeSegmentalLocalizer):
    """Inherited XLS-R reader with one anchor-free native-160 decision path."""

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
        native_interval_hidden: int = 64,
        native_interval_use_context: bool = False,
        native_interval_context_hidden: int = 32,
        native_interval_context_layers: int = 2,
        native_interval_use_boundary_refine: bool = False,
        native_interval_extent_scale: float = 32.0,
        native_interval_intersection_temperature: float = 0.5,
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
        self.native_pool = NativeIntervalClassifier(
            self.out2.in_features,
            **pool_args,
            interval_hidden=native_interval_hidden,
            interval_use_context=native_interval_use_context,
            interval_context_hidden=native_interval_context_hidden,
            interval_context_layers=native_interval_context_layers,
            interval_use_boundary_refine=native_interval_use_boundary_refine,
            interval_extent_scale=native_interval_extent_scale,
            interval_intersection_temperature=native_interval_intersection_temperature,
        )
        self.native_pool.load_state_dict(inherited, strict=False)
