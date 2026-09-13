"""A single trained frame witness readout for the native 160 ms grid."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spanmark.models import CoarseTemporalResidualBlock, PositionAnchorLocalizer


def power_pool_bonafide_logits(
    frame_logits: torch.Tensor,
    valid: torch.Tensor,
    power: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sole bonafide logits from positive-power spoof pooling.

    The final two dimensions of ``frame_logits`` are frames and the classes
    (spoof, bonafide). Both log-sum-exp reductions exclude invalid frames.
    FP64 inputs remain FP64 so this arithmetic can be checked independently of
    the classifier's FP32 runtime. Invalid logits, including NaN/Inf padding,
    are removed before any nonlinear operation.
    """
    if frame_logits.ndim < 2 or frame_logits.shape[-1] != 2:
        raise ValueError("witness logits must end in (frames, 2)")
    if valid.shape != frame_logits.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("witness validity must be a matching boolean frame mask")
    if valid.device != frame_logits.device:
        raise ValueError("witness logits and validity must share a device")
    if not frame_logits.is_floating_point():
        raise ValueError("witness logits must be floating point")
    if frame_logits.shape[-2] < 1:
        raise ValueError("witness pooling requires at least one frame slot")

    logits = frame_logits if frame_logits.dtype == torch.float64 else frame_logits.float()
    logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
    exponent = torch.as_tensor(power, dtype=logits.dtype, device=logits.device)
    if exponent.numel() != 1:
        raise ValueError("witness power must be a scalar")
    exponent = exponent.reshape(())
    log_prob = F.log_softmax(logits, dim=-1)
    log_spoof, log_bonafide = log_prob.unbind(dim=-1)

    block_valid = valid.any(dim=-1)
    # Empty blocks use finite dummy terms in both reductions; masking the
    # result alone would still backpropagate through (-inf) - (-inf).
    reduction_mask = valid | ~block_valid.unsqueeze(-1)
    numerator = (exponent * log_spoof + log_bonafide).masked_fill(
        ~reduction_mask, -torch.inf
    )
    denominator = ((exponent + 1.0) * log_spoof).masked_fill(
        ~reduction_mask, -torch.inf
    )
    pooled = torch.logsumexp(numerator, dim=-1) - torch.logsumexp(denominator, dim=-1)

    # For a singleton the powers cancel exactly. Use its literal class-logit
    # difference rather than subtracting two potentially large reductions.
    singleton = valid.sum(dim=-1) == 1
    singleton_score = (logits[..., 1] - logits[..., 0]).masked_fill(~valid, 0.0).sum(dim=-1)
    score = torch.where(singleton, singleton_score, pooled)
    return score.masked_fill(~block_valid, 0.0), block_valid


class NativeWitnessClassifier(nn.Module):
    """One local block followed by a two-class frame head and learned power.

    The registered 512D/64D configuration has 13 tensors and 72,707 parameters.
    It contains no attention or summary classifier. Context resets at every
    eight-frame boundary, matching the independently sampled training blocks.
    """

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160,),
        block_layers: int = 1,
        block_bottleneck: int = 64,
        dropout: float = 0.1,
        initial_power: float = 1.0,
    ):
        super().__init__()
        if tuple(resolutions_ms) != (160,):
            raise ValueError("the witness classifier supplies native 160 ms only")
        if block_layers != 1:
            raise ValueError("the registered witness classifier requires one local block")
        if state_dim < 1 or block_bottleneck < 1:
            raise ValueError("witness state and bottleneck widths must be positive")
        if not math.isfinite(initial_power) or initial_power <= 0:
            raise ValueError("initial witness power must be finite and positive")

        self.state_dim = int(state_dim)
        self.resolutions_ms = (160,)
        self.block_sizes = {160: 8}
        self.block_trunk = nn.ModuleList([
            CoarseTemporalResidualBlock(
                self.state_dim,
                bottleneck=block_bottleneck,
                dilation=1,
                dropout=dropout,
            )
        ])
        self.frame_head = nn.Linear(self.state_dim, 2)
        # The equivalent expression stays finite even for a large positive
        # initial power; at the registered value 1 it is log(expm1(1)).
        inverse_softplus = initial_power + math.log(-math.expm1(-initial_power))
        self.rho = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))

    def power(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if states.ndim != 3 or valid.shape != states.shape[:2]:
            raise ValueError("witness classifier expects states (B,T,D) and valid (B,T)")
        if states.shape[-1] != self.state_dim or valid.dtype != torch.bool:
            raise ValueError("witness classifier state width or valid dtype is wrong")
        if states.device != valid.device:
            raise ValueError("witness states and validity must share a device")

        batch, frames, _ = states.shape
        if frames == 0:
            return (
                {160: states.new_zeros((batch, 0), dtype=torch.float32)},
                {160: valid.new_zeros((batch, 0))},
            )
        with torch.autocast(device_type=states.device.type, enabled=False):
            # Multiplication would preserve NaNs in padded states and allow
            # them to enter neighboring valid frames through the convolution.
            encoded = states.float().masked_fill(~valid.unsqueeze(-1), 0.0)
            pad = (-frames) % 8
            encoded = F.pad(encoded, (0, 0, 0, pad))
            block_mask = F.pad(valid, (0, pad), value=False).reshape(batch, -1, 8)
            local = encoded.reshape(-1, 8, self.state_dim)
            local_mask = block_mask.reshape(-1, 8)
            for layer in self.block_trunk:
                local = layer(local, local_mask)
            frame_logits = self.frame_head(local).reshape(batch, -1, 8, 2)
            scores, block_valid = power_pool_bonafide_logits(
                frame_logits, block_mask, self.power()
            )
        return {160: scores}, {160: block_valid}


class NativeWitnessLocalizer(PositionAnchorLocalizer):
    """The inherited anchor reader with one native witness decision path."""

    def __init__(
        self,
        *args,
        native_resolutions_ms: tuple[int, ...] = (160,),
        native_dropout: float = 0.1,
        native_block_layers: int = 1,
        native_block_bottleneck: int = 64,
        native_initial_power: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.native_pool = NativeWitnessClassifier(
            self.out2.in_features,
            resolutions_ms=tuple(native_resolutions_ms),
            block_layers=native_block_layers,
            block_bottleneck=native_block_bottleneck,
            dropout=native_dropout,
            initial_power=native_initial_power,
        )

    def _head_forward(self, feats, valid, n_segments, seg_max):
        base_logits = super()._head_forward(feats, valid, n_segments, seg_max)
        with torch.autocast(device_type=feats.device.type, enabled=False):
            native_scores, native_valid = self.native_pool(
                self._aux["y2"].float(), valid
            )
        self._aux["native_scores"] = native_scores
        self._aux["native_valid"] = native_valid
        return base_logits

    @torch.inference_mode()
    def score_with_native(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor,
        n_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        logits = self(wav, lengths, n_segments)
        fine = logits[..., 1] - logits[..., 0]
        native = self._aux.get("native_scores")
        if not isinstance(native, dict) or set(native) != {160}:
            raise RuntimeError("witness classifier did not produce only native 160 ms scores")
        return fine, native
