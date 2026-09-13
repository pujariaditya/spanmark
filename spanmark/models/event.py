"""Exact all-bonafide versus any-spoof events in an eight-frame native block.

The inherited summary score supplies the all-bonafide event potential. Learned
unary and optional adjacent-pair potentials distribute the any-spoof event's
mass over its possible frame configurations. Both are parts of one native
classifier and are jointly trained; no independent model scores are combined.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spanmark.models import NativePooledClassifier, NativePooledLocalizer


def event_pattern_partition(
    unary: torch.Tensor,
    pair: torch.Tensor | None,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return configuration energies, valid lengths, and log Z(any spoof).

    ``unary`` and boolean ``valid`` end in eight frame slots; ``pair``, when
    supplied, ends in ``(7, 4)``. Bit t of the configuration index is one for
    spoof at frame t. Pair channels use ``2 * left_bit + right_bit``; channel
    zero is therefore the bonafide/bonafide reference. Right-padded tails
    allow only configurations whose remaining bits are zero.

    The all-bonafide configuration has energy zero exactly. Unsupported
    configurations have energy -inf. An empty block uses that zero-energy
    configuration as a finite dummy reduction, returning log Z = 0. FP64
    inputs retain FP64 for independent numerical and gradient checks; normal
    model arithmetic is FP32. Invalid NaN/Inf padding is removed before it
    enters a product, pair centering, or nonlinear reduction.
    """
    if unary.ndim < 1 or unary.shape[-1] != 8 or not unary.is_floating_point():
        raise ValueError("event unary potentials must be floating point with eight slots")
    if valid.shape != unary.shape or valid.dtype != torch.bool:
        raise ValueError("event validity must match unary potentials and be boolean")
    if valid.device != unary.device:
        raise ValueError("event potentials and validity must share a device")
    if pair is not None:
        if pair.shape != unary.shape[:-1] + (7, 4) or not pair.is_floating_point():
            raise ValueError("event pair potentials must end in (7, 4)")
        if pair.device != unary.device:
            raise ValueError("event unary and pair potentials must share a device")

    lengths = valid.sum(dim=-1)
    slots = torch.arange(8, device=unary.device)
    if not torch.equal(valid, slots < lengths.unsqueeze(-1)):
        raise ValueError("event blocks require valid frames followed by right padding")

    values = unary if unary.dtype == torch.float64 else unary.float()
    values = values.masked_fill(~valid, 0.0)
    indices = torch.arange(256, device=unary.device)
    bits = (indices[:, None] >> slots) & 1
    energies = values @ bits.to(values.dtype).T
    if pair is not None:
        adjacent = valid[..., :-1] & valid[..., 1:]
        pair_values = pair.to(values.dtype).masked_fill(~adjacent.unsqueeze(-1), 0.0)
        pair_values = pair_values - pair_values[..., :1]
        pair_indices = 2 * bits[:, :-1] + bits[:, 1:]
        selection = torch.nn.functional.one_hot(pair_indices, 4).to(values.dtype)
        energies = energies + pair_values.flatten(-2) @ selection.flatten(-2).T

    supported = indices < (1 << lengths.unsqueeze(-1))
    energies = energies.masked_fill(~supported, -torch.inf)
    any_spoof = supported & (indices > 0)
    reduction_mask = any_spoof | ((lengths == 0).unsqueeze(-1) & (indices == 0))
    log_z_any = torch.logsumexp(energies.masked_fill(~reduction_mask, -torch.inf), dim=-1)
    return energies, lengths, log_z_any


class NativeEventClassifier(NativePooledClassifier):
    """Seed-compatible summary potential with a learned block event field."""

    def __init__(
        self, state_dim: int, resolutions_ms: tuple[int, ...] = (160,),
        *args, event_hidden: int = 64, event_pairwise: bool = True, **kwargs,
    ):
        if tuple(resolutions_ms) != (160,):
            raise ValueError("the event classifier supplies native 160 ms only")
        if event_hidden < 1:
            raise ValueError("event hidden width must be positive")
        if type(event_pairwise) is not bool:
            raise ValueError("event_pairwise must be boolean")
        super().__init__(state_dim, resolutions_ms, *args, **kwargs)
        self.event_hidden = int(event_hidden)
        self.event_pairwise = event_pairwise

        def potential(input_dim: int, output_dim: int) -> nn.Sequential:
            result = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.event_hidden),
                nn.GELU(),
                nn.Linear(self.event_hidden, output_dim),
            )
            nn.init.zeros_(result[-1].weight)
            nn.init.zeros_(result[-1].bias)
            return result

        self.event_unary = potential(self.state_dim, 1)
        if event_pairwise:
            self.event_pair = potential(2 * self.state_dim, 4)
        self._event_aux: dict[str, torch.Tensor] = {}

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        self._event_aux = {}
        if states.ndim != 3 or valid.shape != states.shape[:2]:
            raise ValueError("event classifier expects states (B,T,D) and valid (B,T)")
        if states.shape[-1] != self.state_dim or valid.dtype != torch.bool:
            raise ValueError("event classifier state width or valid dtype is wrong")
        if states.device != valid.device:
            raise ValueError("event states and validity must share a device")

        with torch.autocast(device_type=states.device.type, enabled=False):
            # Mask before the inherited convolutions: NaN * 0 is still NaN.
            encoded = states.float().masked_fill(~valid.unsqueeze(-1), 0.0)
            if states.shape[1] == 0 or states.shape[0] == 0:
                batch = states.shape[0]
                blocks = (states.shape[1] + 7) // 8
                unary = encoded.sum(dim=-1).new_zeros((batch, blocks, 8))
                block_mask = valid.new_zeros((batch, blocks, 8))
                energies, lengths, log_z = event_pattern_partition(unary, None, block_mask)
                self._event_aux.update(
                    pattern_energies=energies, lengths=lengths,
                    logZ_any=log_z, block_valid=block_mask, frame_logits=unary,
                )
                result = ({160: log_z}, {160: block_mask.any(dim=-1)})
            else:
                result = super().forward(encoded, valid)
        self._event_aux["frame_valid"] = valid
        return result

    def _refine_native_score(
        self, ms: int, block_states: torch.Tensor,
        block_mask: torch.Tensor, score: torch.Tensor,
    ) -> torch.Tensor:
        unary = self.event_unary(block_states).squeeze(-1)
        pair = None
        if self.event_pairwise:
            adjacent = torch.cat([block_states[..., :-1, :], block_states[..., 1:, :]], dim=-1)
            pair = self.event_pair(adjacent)
        energies, lengths, log_z = event_pattern_partition(unary, pair, block_mask)
        self._event_aux.update(
            pattern_energies=energies, lengths=lengths, frame_logits=unary,
            logZ_any=log_z, block_valid=block_mask,
        )
        log_pattern_count = ((1 << lengths) - 1).clamp_min(1).to(log_z.dtype).log()
        # Subtract the centered log partition so zero potentials are literally
        # an identity, avoiding cancellation between a seed score and log 255.
        return score - (log_z - log_pattern_count)

    def conditional_pattern_nll(
        self, frame_labels: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mean log Z(any spoof) - E(gold) over spoof-containing blocks.

        Labels use the training convention 0=spoof, 1=bonafide. Padded labels
        are ignored. The inherited final summary score is absent from this
        conditional objective, so its classifier and attention parameters
        receive no gradient from this loss. Shared local states can move.
        """
        if "frame_valid" not in self._event_aux:
            raise RuntimeError("call the event classifier before conditional_pattern_nll")
        previous_valid = self._event_aux["frame_valid"]
        if frame_labels.shape != previous_valid.shape or valid.shape != previous_valid.shape:
            raise ValueError("conditional event labels must match the preceding forward")
        if valid.dtype != torch.bool or valid.device != previous_valid.device:
            raise ValueError("conditional event validity must be a matching boolean mask")
        if frame_labels.device != previous_valid.device or not torch.equal(valid, previous_valid):
            raise ValueError("conditional event loss must use the preceding forward's validity")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid event frame labels must be 0=spoof or 1=bonafide")

        energies = self._event_aux["pattern_energies"]
        log_z = self._event_aux["logZ_any"]
        if frame_labels.numel() == 0:
            return log_z.sum()
        spoof_bits = self._blockify((frame_labels == 0) & valid, 8, 0).long()
        powers = 1 << torch.arange(8, device=frame_labels.device)
        gold_index = (spoof_bits * powers).sum(dim=-1)
        gold_energy = energies.gather(-1, gold_index.unsqueeze(-1)).squeeze(-1)
        spoof_blocks = gold_index > 0
        nll = (log_z - gold_energy).masked_fill(~spoof_blocks, 0.0)
        return nll.sum() / spoof_blocks.sum().clamp_min(1)

    def frame_auxiliary_loss(
        self,
        frame_labels: torch.Tensor,
        valid: torch.Tensor,
        *,
        kind: str,
        focal_gamma: float = 2.0,
        boundary_mask: torch.Tensor | None = None,
        partial_focal_weight: float = 0.5,
    ) -> torch.Tensor:
        """Strong-label loss on the event unary logits, never the fine route.

        Event unary energy represents the spoof bit, so the auxiliary target is
        ``1 - frame_labels``. ``partial`` follows HarmoNet's qualitative form:
        normalized focal loss on all valid frames plus normalized BCE on a
        supplied transition neighborhood. The caller constructs that mask from
        whole-utterance labels so jittered block edges do not hide transitions.
        """
        if "frame_valid" not in self._event_aux or "frame_logits" not in self._event_aux:
            raise RuntimeError("call the event classifier before frame_auxiliary_loss")
        previous_valid = self._event_aux["frame_valid"]
        if frame_labels.shape != previous_valid.shape or valid.shape != previous_valid.shape:
            raise ValueError("strong event labels must match the preceding forward")
        if valid.dtype != torch.bool or valid.device != previous_valid.device:
            raise ValueError("strong event validity must be a matching boolean mask")
        if frame_labels.device != previous_valid.device or not torch.equal(valid, previous_valid):
            raise ValueError("strong event loss must use the preceding forward's validity")
        if bool(((frame_labels != 0) & (frame_labels != 1) & valid).any()):
            raise ValueError("valid strong event labels must be 0=spoof or 1=bonafide")
        if kind not in ("none", "bce", "focal", "partial"):
            raise ValueError(f"unknown strong event auxiliary: {kind}")
        if not isinstance(focal_gamma, (int, float)) or focal_gamma < 0:
            raise ValueError("focal gamma must be a nonnegative scalar")
        if not isinstance(partial_focal_weight, (int, float)) or not 0 <= partial_focal_weight <= 1:
            raise ValueError("partial focal weight must lie in [0, 1]")
        if boundary_mask is not None and (
                boundary_mask.shape != valid.shape or boundary_mask.dtype != torch.bool
                or boundary_mask.device != valid.device):
            raise ValueError("boundary mask must be a matching boolean tensor")
        if kind == "partial" and boundary_mask is None:
            raise ValueError("partial event loss requires a boundary mask")

        logits = self._event_aux["frame_logits"]
        blocked_valid = self._blockify(valid, 8, False)
        if logits.shape != blocked_valid.shape:
            raise RuntimeError("event unary logits do not match the preceding validity")
        if not bool(blocked_valid.any()):
            return logits.sum() * 0.0
        target = self._blockify((frame_labels == 0) & valid, 8, False).to(logits.dtype)
        per_frame = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = per_frame[blocked_valid].mean()
        if kind == "none":
            return logits.sum() * 0.0
        if kind == "bce":
            return bce
        probability_true = torch.exp(-per_frame)
        focal_per_frame = (1.0 - probability_true).pow(float(focal_gamma)) * per_frame
        focal = focal_per_frame[blocked_valid].mean()
        if kind == "focal":
            return focal
        blocked_boundary = self._blockify(boundary_mask & valid, 8, False) & blocked_valid
        if not bool(blocked_boundary.any()):
            return focal
        boundary_bce = per_frame[blocked_boundary].mean()
        mix = float(partial_focal_weight)
        return mix * focal + (1.0 - mix) * boundary_bce


class NativeEventLocalizer(NativePooledLocalizer):
    """Inherited fine reader with one event-field native decision path."""

    def __init__(
        self, *args, native_resolutions_ms: tuple[int, ...] = (160,),
        native_attention_hidden: int = 64, native_classifier_hidden: int = 128,
        native_dropout: float = 0.1, native_trunk_layers: int = 0,
        native_trunk_bottleneck: int = 64, native_zero_output_init: bool = True,
        native_summary_mode: str = "meanstd", native_block_layers: int = 0,
        native_block_bottleneck: int = 64, native_event_hidden: int = 64,
        native_event_pairwise: bool = True, **kwargs,
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
        )
        super().__init__(*args, **{"native_" + key: value for key, value in pool_args.items()}, **kwargs)
        inherited = self.native_pool.state_dict()
        self.native_pool = NativeEventClassifier(
            self.out2.in_features, **pool_args,
            event_hidden=native_event_hidden, event_pairwise=native_event_pairwise,
        )
        # Preserve constructor-level identity as well as checkpoint transfer.
        self.native_pool.load_state_dict(inherited, strict=False)
