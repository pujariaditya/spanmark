"""Frame-level localizer: one score per 20 ms segment.

Two frontends:
  * mfcc  -- the seed's 10 ms MFCC stack (kept so the contract tests and the
             baseline checkpoint still load).
  * xlsr  -- wav2vec2 XLS-R 300m, optionally FINE-TUNED, at 50 Hz which is
             exactly the 20 ms scoring grid. Frame i of XLS-R covers samples
             [320 i, 320 i + 400), i.e. segment i plus 5 ms of look-ahead, so
             frames map onto segments by index: no interpolation is needed if
             the waveform is right-padded so that at least n_segments frames
             come out. That is what `XLSRLocalizer` does.

Alignment is the one thing that must never drift: a one-segment shift keeps
the EER plausible while destroying the localization it claims to measure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from spanmark.data import HOP, SAMPLE_RATE
from spanmark.models.conditional_layers import ConditionalLayerGate

FRONTENDS = ("mfcc", "xlsr")


class MFCCFrontend(nn.Module):
    out_dim = 180
    frame_hz = 100.0          # 10 ms hop

    @staticmethod
    def valid_frames(lengths: torch.Tensor) -> torch.Tensor:
        """Frames of real audio, excluding batch padding (center=True, hop 160)."""
        return torch.clamp(lengths // 160 + 1, min=1)

    def __init__(self, n_mfcc: int = 60, f_min: float = 20.0, f_max: float | None = None):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80,
                       "center": True, "f_min": f_min, "f_max": f_max},
        )
        self.delta = torchaudio.transforms.ComputeDeltas()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.mfcc(wav)
        d1 = self.delta(x)
        d2 = self.delta(d1)
        return torch.cat([x, d1, d2], dim=1).transpose(1, 2)     # (B, T, D)


class MultiScaleTemporalAdapter(nn.Module):
    """Zero-output multi-scale residual adapter for a frozen SSL state.

    The bottleneck follows MultiConvAdapter's temporal prior: split channels
    across several depthwise kernels, then fuse them with a residual width-3
    convolution.  The final projection is initialized to exact zero, so adding
    an adapter cannot perturb a pretrained parent before its first update.
    """

    def __init__(self, input_dim: int = 1024, bottleneck: int = 64,
                 kernels: tuple[int, ...] = (3, 7, 15, 23)):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels or any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("adapter kernels must be positive odd integers")
        if bottleneck <= 0 or bottleneck % len(kernels):
            raise ValueError("adapter bottleneck must be positive and divisible by kernel count")
        branch_dim = bottleneck // len(kernels)
        self.kernels = kernels
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck)
        self.temporal = nn.ModuleList([
            nn.Conv1d(branch_dim, branch_dim, kernel_size=k, padding=k // 2,
                      groups=branch_dim)
            for k in kernels
        ])
        self.mix = nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1)
        self.up = nn.Linear(bottleneck, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        mask = None
        if valid is not None:
            # Sanitise before LayerNorm and temporal convolution: multiplying
            # NaN padding by zero after the projection does not remove NaNs,
            # and a convolution can then leak them into neighbouring valid
            # frames.  Keep the residual `x` untouched so valid outputs remain
            # exactly identical under the zero-output initialisation.
            x_adapter = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        else:
            x_adapter = x
        h = F.gelu(self.down(self.norm(x_adapter))).transpose(1, 2)  # (B, D', T)
        if valid is not None:
            mask = valid.unsqueeze(1).to(h.dtype)
            h = h * mask
        chunks = h.chunk(len(self.temporal), dim=1)
        h = torch.cat([conv(chunk) for conv, chunk in zip(self.temporal, chunks)], dim=1)
        h = F.gelu(h)
        if mask is not None:
            h = h * mask
        h = h + self.mix(h)
        if mask is not None:
            h = h * mask
        delta = self.up(h.transpose(1, 2))
        if valid is not None:
            delta = delta * valid.unsqueeze(-1).to(delta.dtype)
        return x + delta


class XLSRFrontend(nn.Module):
    """wav2vec2-style SSL encoder. `freeze=False` fine-tunes the transformer
    (the conv feature extractor stays frozen; that is the usual recipe).
    `layer=None` learns a softmax-weighted sum over all hidden states."""

    out_dim = 1024
    frame_hz = 50.0           # 20 ms hop -- already the scoring grid

    def valid_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.model._get_feat_extract_output_lengths(lengths), min=1)

    def __init__(self, name: str = "facebook/wav2vec2-xls-r-300m",
                 revision: str | None = None, layer: int | None = None,
                 freeze: bool = True, n_layers_keep: int | None = None,
                 layerdrop: float = 0.05, layer_range: tuple[int, int] | None = None,
                 adapter_dim: int = 0, adapter_kernels: tuple[int, ...] = (3, 7, 15, 23),
                 adapter_layers: tuple[int, ...] | None = None,
                 unfreeze_top: int = 0, conditional_layer_gate: bool = False,
                 conditional_logit_bound: float = 1.0):
        """layer_range=(lo, hi): the learned softmax mixture is restricted to hidden states
        lo..hi inclusive (others get -inf); the encoder still runs fully."""
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            name, revision=revision
        )  # wav2vec2 / XLS-R / HuBERT / MMS: same API
        if unfreeze_top < 0:
            raise ValueError("unfreeze_top must be non-negative")
        if unfreeze_top and not freeze:
            raise ValueError(
                "unfreeze_top requires freeze=True (freeze the base, then select top blocks)"
            )
        cfg = self.model.config
        cfg.apply_spec_augment = False       # SpecAugment masks would corrupt frame labels
        cfg.layerdrop = layerdrop if not freeze else 0.0
        if n_layers_keep is not None and n_layers_keep < len(self.model.encoder.layers):
            self.model.encoder.layers = self.model.encoder.layers[:n_layers_keep]
            cfg.num_hidden_layers = n_layers_keep
        n_blocks = len(self.model.encoder.layers)
        if unfreeze_top > n_blocks:
            raise ValueError(
                f"cannot unfreeze {unfreeze_top} top blocks from an encoder with {n_blocks}"
            )
        self.n_hidden = n_blocks + 1
        self.layer = layer
        # Selective adaptation keeps encoder stochastic layers disabled while
        # allowing autograd through the declared suffix. ``frozen`` controls
        # no_grad; ``encoder_eval`` controls module mode.
        self.encoder_eval = bool(freeze)
        self.frozen = bool(freeze and unfreeze_top == 0)
        self.unfreeze_top = int(unfreeze_top)
        self.out_dim = cfg.hidden_size
        if adapter_dim and layer is not None:
            raise ValueError("temporal adapters require the learned all-state mixture")
        if adapter_layers is None:
            adapter_layers = tuple(range(self.n_hidden))
        else:
            adapter_layers = tuple(sorted(set(int(i) for i in adapter_layers)))
        if any(i < 0 or i >= self.n_hidden for i in adapter_layers):
            raise ValueError(f"adapter layer index outside 0..{self.n_hidden - 1}")
        if type(conditional_layer_gate) is not bool:
            raise ValueError("conditional_layer_gate must be boolean")
        if conditional_layer_gate and (layer is not None or adapter_dim or layer_range is not None):
            raise ValueError("conditional gate requires the unmodified all-layer mixture")
        self.conditional_gate = (ConditionalLayerGate(self.out_dim, self.n_hidden,
                                                     conditional_logit_bound)
                                 if conditional_layer_gate else None)
        self.adapter_dim = int(adapter_dim)
        self.adapter_layers = adapter_layers if adapter_dim else ()
        self.adapters = nn.ModuleDict({
            str(i): MultiScaleTemporalAdapter(self.out_dim, self.adapter_dim,
                                              tuple(adapter_kernels))
            for i in self.adapter_layers
        })
        if layer is None:
            self.layer_weights = nn.Parameter(torch.zeros(self.n_hidden))
        self.layer_range = tuple(layer_range) if layer_range is not None else None
        if self.layer_range is not None:
            mask = torch.full((self.n_hidden,), float("-inf"))
            mask[self.layer_range[0]: self.layer_range[1] + 1] = 0.0
            self.register_buffer("layer_mask", mask, persistent=False)
        self.model.feature_extractor._freeze_parameters()
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            if unfreeze_top:
                for block in self.model.encoder.layers[-unfreeze_top:]:
                    for p in block.parameters():
                        p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_eval:
            self.model.eval()
        return self

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor | None = None,
                manifest_frames: torch.Tensor | None = None) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        attn = None
        if lengths is not None:
            attn = (torch.arange(wav.shape[1], device=wav.device)[None, :]
                    < lengths.to(wav.device)[:, None]).long()
        with ctx:
            out = self.model(wav, attention_mask=attn, output_hidden_states=True)
        if self.layer is None:
            states = list(out.hidden_states)
            if self.adapters:
                valid = None
                # The downstream contract is defined by the manifest grid, which
                # can extend one or two cells beyond XLS-R's acoustic receptive-
                # field count.  Adapt every frame the reader will score, while
                # still masking batch padding beyond that requested count.
                if manifest_frames is not None:
                    n_valid = manifest_frames.to(states[0].device)
                elif lengths is not None:
                    n_valid = self.valid_frames(lengths.to(states[0].device))
                if manifest_frames is not None or lengths is not None:
                    n_valid = n_valid.clamp(max=states[0].shape[1])
                    valid = (torch.arange(states[0].shape[1], device=states[0].device)[None, :]
                             < n_valid[:, None])
                states = [self.adapters[str(i)](state, valid) if str(i) in self.adapters else state
                          for i, state in enumerate(states)]
            hs = torch.stack(states, dim=0)                         # (L, B, T, D)
            lw = self.layer_weights + self.layer_mask if self.layer_range is not None else self.layer_weights
            w = torch.softmax(lw, dim=0).to(hs.dtype)
            inherited = (hs * w[:, None, None, None]).sum(0)
            # Use getattr for lightweight objects built through __new__ in
            # contract tests.  A class-level ``conditional_gate = None`` cannot
            # provide this fallback: it shadows nn.Module's lookup in _modules
            # and silently makes a registered nonzero gate unreachable.
            conditional_gate = getattr(self, "conditional_gate", None)
            if conditional_gate is not None:
                return conditional_gate(hs, lw, inherited)
            return inherited
        if self.layer >= self.n_hidden - 1:
            return out.last_hidden_state
        return out.hidden_states[self.layer]


class Localizer(nn.Module):
    """frontend -> temporal conv -> one logit pair per 20 ms segment (the seed head)."""

    def __init__(self, frontend: str = "mfcc", hidden: int = 256, dropout: float = 0.2, **kw):
        super().__init__()
        if frontend not in FRONTENDS:
            raise ValueError(f"frontend must be one of {FRONTENDS}, got {frontend!r}")
        self.frontend_name = frontend
        self.frontend = MFCCFrontend(**kw) if frontend == "mfcc" else XLSRFrontend(**kw)
        d = self.frontend.out_dim
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, hidden, 5, padding=2), nn.GELU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(), nn.BatchNorm1d(hidden),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(hidden, 2, 1)

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor,
                n_segments: torch.Tensor) -> torch.Tensor:
        """Return (B, S_max, 2) logits on the 20 ms grid (see seed docstring)."""
        feats = self.norm(self.frontend(wav)).transpose(1, 2)     # (B, D, T)
        vf = self.frontend.valid_frames(lengths.to(feats.device))
        vf = torch.clamp(vf, max=feats.shape[-1])
        keep = (torch.arange(feats.shape[-1], device=feats.device)[None, :]
                < vf[:, None]).unsqueeze(1)
        feats = feats * keep

        h = self.drop(self.conv(feats))
        logits = self.head(h)                                     # (B, 2, T)
        seg_max = int(n_segments.max())
        out = logits.new_zeros(logits.shape[0], 2, seg_max)
        for i in range(logits.shape[0]):
            n = int(n_segments[i])
            piece = logits[i : i + 1, :, : int(vf[i])]
            out[i : i + 1, :, :n] = F.interpolate(
                piece, size=n, mode="linear", align_corners=False
            )
        return out.transpose(1, 2)                                # (B, S, 2)

    @torch.inference_mode()
    def score(self, wav: torch.Tensor, lengths: torch.Tensor,
              n_segments: torch.Tensor) -> torch.Tensor:
        """logit(bonafide) - logit(spoof) per segment. HIGHER = MORE BONAFIDE."""
        out = self(wav, lengths, n_segments)
        return out[..., 1] - out[..., 0]


class XLSRLocalizer(nn.Module):
    """Fine-tunable XLS-R -> (conv, BiLSTM) head -> logits, index-aligned to the 20 ms grid.

    The waveform is right-padded by one receptive field so XLS-R emits at least
    n_segments frames for every utterance; frame i is then segment i. Frames
    past each utterance's own segment count are ignored (loss mask / slicing),
    so a score depends only on its own utterance, never on the batch buffer.
    """

    RF = 400                                                       # XLS-R receptive field

    def __init__(self, name: str = "facebook/wav2vec2-xls-r-300m",
                 revision: str | None = None, layer: int | None = None,
                 freeze: bool = False, n_layers_keep: int | None = None, hidden: int = 256,
                 dropout: float = 0.1, head: str = "bilstm", layerdrop: float = 0.05,
                 feat_norm: str = "none", wav_norm: bool = False,
                 layer_range: tuple[int, int] | None = None,
                 adapter_dim: int = 0, adapter_kernels: tuple[int, ...] = (3, 7, 15, 23),
                 adapter_layers: tuple[int, ...] | None = None,
                 unfreeze_top: int = 0, conditional_layer_gate: bool = False,
                 conditional_logit_bound: float = 1.0):
        """feat_norm -- per-UTTERANCE statistics of the SSL features, computed
        over that utterance's own valid frames only:
            none     : LayerNorm(x)                          (absolute features)
            mean     : LayerNorm(x - mean_utt(x))            (channel/speaker offset removed)
            meanstd  : LayerNorm((x - mean_utt) / std_utt)
            concat   : [LayerNorm(x) ; LayerNorm(x - mean_utt(x))]  (absolute + contrast-with-context)
        The contrast variants score a segment by how much it differs from the
        rest of ITS utterance, which is a generator-agnostic cue for an inserted
        span and is what an unseen corpus can still offer."""
        super().__init__()
        self.frontend_name = "xlsr"
        self.frontend = XLSRFrontend(name, revision=revision, layer=layer, freeze=freeze,
                                     n_layers_keep=n_layers_keep, layerdrop=layerdrop,
                                     layer_range=layer_range, adapter_dim=adapter_dim,
                                     adapter_kernels=adapter_kernels,
                                     adapter_layers=adapter_layers,
                                     unfreeze_top=unfreeze_top,
                                     conditional_layer_gate=conditional_layer_gate,
                                     conditional_logit_bound=conditional_logit_bound)
        d = self.frontend.out_dim
        if feat_norm not in ("none", "mean", "meanstd", "concat"):
            raise ValueError(feat_norm)
        self.feat_norm = feat_norm
        # XLS-R was pretrained with do_normalize=True (zero-mean, unit-variance waveform
        # per utterance). Off by default only so that checkpoints trained before the flag
        # existed keep loading; every new model should set it.
        self.wav_norm = wav_norm
        self.norm = nn.LayerNorm(d)
        if feat_norm == "concat":
            self.norm2 = nn.LayerNorm(d)
        in_d = 2 * d if feat_norm == "concat" else d
        self.proj = nn.Sequential(nn.Linear(in_d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head_type = head
        if head == "bilstm":
            self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=dropout)
            out_d = 2 * hidden
        elif head == "conv":
            self.rnn = None
            self.conv = nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU())
            out_d = hidden
        else:
            raise ValueError(head)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(out_d, 2)

    def param_groups(self, lr_ssl: float, lr_head: float):
        ssl = [p for p in self.frontend.model.parameters() if p.requires_grad]
        head = [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("frontend.model.")]
        groups = [{"params": head, "lr": lr_head}]
        if ssl:
            groups.append({"params": ssl, "lr": lr_ssl})
        return groups

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor,
                n_segments: torch.Tensor) -> torch.Tensor:
        seg_max = int(n_segments.max())
        need = seg_max * HOP + self.RF
        if wav.shape[1] < need:
            wav = F.pad(wav, (0, need - wav.shape[1]))
        # attention mask over REAL samples so padding never feeds the encoder
        lengths = lengths.to(wav.device)
        if self.wav_norm:
            keep = (torch.arange(wav.shape[1], device=wav.device)[None, :] < lengths[:, None]).to(wav.dtype)
            n = lengths.clamp(min=1).unsqueeze(1).to(wav.dtype)
            m = (wav * keep).sum(1, keepdim=True) / n
            v = (((wav - m) * keep) ** 2).sum(1, keepdim=True) / n
            wav = (wav - m) * torch.rsqrt(v + 1e-7) * keep
        feats = self.frontend(wav, lengths, n_segments)            # (B, T, D)
        T = feats.shape[1]
        if T < seg_max:
            feats = F.pad(feats, (0, 0, 0, seg_max - T))
        feats = feats[:, :seg_max]
        # valid = frames whose window starts inside the real audio == segments
        valid = (torch.arange(seg_max, device=wav.device)[None, :]
                 < n_segments.to(wav.device)[:, None])
        return self._head_forward(feats, valid, n_segments, seg_max)

    def _head_forward(self, feats: torch.Tensor, valid: torch.Tensor,
                      n_segments: torch.Tensor, seg_max: int) -> torch.Tensor:
        if self.feat_norm == "none":
            x = self.norm(feats)
        else:
            v = valid.unsqueeze(-1).to(feats.dtype)
            cnt = v.sum(1, keepdim=True).clamp(min=1.0)
            mu = (feats * v).sum(1, keepdim=True) / cnt
            centred = feats - mu
            if self.feat_norm == "mean":
                x = self.norm(centred)
            elif self.feat_norm == "meanstd":
                var = ((centred * v) ** 2).sum(1, keepdim=True) / cnt
                x = self.norm(centred * torch.rsqrt(var + 1e-5))
            else:
                x = torch.cat([self.norm(feats), self.norm2(centred)], dim=-1)
        x = self.proj(x) * valid.unsqueeze(-1)
        if self.rnn is not None:
            lens = n_segments.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True,
                                                       enforce_sorted=False)
            y, _ = self.rnn(packed)
            y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=seg_max)
        else:
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.out(self.drop(y))                              # (B, S, 2)

    @torch.inference_mode()
    def score(self, wav: torch.Tensor, lengths: torch.Tensor,
              n_segments: torch.Tensor) -> torch.Tensor:
        out = self(wav, lengths, n_segments)
        return out[..., 1] - out[..., 0]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for a (batch, channel, time) tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class UNetResidualBlock(nn.Module):
    """A full-resolution residual block with no recurrent/global state."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv1d(in_channels, out_channels, 1))
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.drop(F.gelu(self.norm1(self.conv1(x))))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + residual) * mask


class BoundaryUNetReader(nn.Module):
    """Boundary-first coarse-to-fine reader over frozen 20 ms SSL features.

    The boundary pyramid is decoded first. Its features and probabilities then
    modulate a separate authenticity decoder at every scale, so the final scores
    are downstream of the learned splice representation rather than merely sharing
    a trunk with an auxiliary boundary head.
    """

    def __init__(self, input_dim: int = 1024, hidden: int = 128,
                 dropout: float = 0.1, probability_gate: bool = False):
        super().__init__()
        channels = (hidden, hidden * 3 // 2, hidden * 2, hidden * 2)
        self.channels = channels
        self.probability_gate = probability_gate
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, channels[0])
        self.encoder = nn.ModuleList([
            UNetResidualBlock(channels[0], channels[0], dropout),
            UNetResidualBlock(channels[1], channels[1], dropout),
            UNetResidualBlock(channels[2], channels[2], dropout),
            UNetResidualBlock(channels[3], channels[3], dropout),
        ])
        self.down_proj = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i + 1], 1) for i in range(3)
        ])

        # Boundary decoder: bottleneck -> 80 ms -> 40 ms -> exact 20 ms seams.
        self.boundary_bottom = UNetResidualBlock(channels[3], channels[3], dropout)
        self.boundary_merge = nn.ModuleList([
            UNetResidualBlock(channels[i] + channels[i + 1], channels[i], dropout)
            for i in (2, 1, 0)
        ])
        self.boundary_out = nn.ModuleList([nn.Conv1d(c, 1, 1) for c in channels])

        # Authenticity decoder is a different path. At each scale its state is
        # affinely conditioned on the already-decoded boundary feature and logit.
        self.class_bottom = UNetResidualBlock(channels[3], channels[3], dropout)
        self.class_merge = nn.ModuleList([
            UNetResidualBlock(channels[i] + channels[i + 1], channels[i], dropout)
            for i in (2, 1, 0)
        ])
        self.boundary_film = nn.ModuleList([
            nn.Conv1d(c + 1, 2 * c, 1) for c in channels
        ])
        self.class_out = nn.Sequential(
            nn.Conv1d(2 * channels[0] + 1, channels[0], 1),
            nn.GELU(), nn.Dropout1d(dropout), nn.Conv1d(channels[0], 2, 1),
        )

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(mask.float(), size=size, mode="nearest")

    @staticmethod
    def _upsample(x: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="linear", align_corners=False)

    def _condition(self, state: torch.Tensor, boundary: torch.Tensor,
                   boundary_logit: torch.Tensor, mask: torch.Tensor, level: int,
                   boundary_scale: float) -> torch.Tensor:
        probability = torch.sigmoid(boundary_logit)
        routed_boundary = boundary * probability if self.probability_gate else boundary
        cond = torch.cat([routed_boundary, probability], dim=1)
        gamma, beta = self.boundary_film[level](cond).chunk(2, dim=1)
        # Bounding gamma prevents an initially noisy seam decoder from erasing the
        # class path, while a zero intervention removes every boundary contribution.
        scale = float(boundary_scale)
        route = scale * probability if self.probability_gate else scale
        state = state * (1.0 + route * 0.25 * torch.tanh(gamma))
        state = state + route * 0.25 * beta
        return state * mask

    def forward(self, feats: torch.Tensor, valid: torch.Tensor,
                boundary_scale: float = 1.0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return segment logits and boundary logits ordered fine-to-coarse."""
        masks = [valid[:, None].to(feats.dtype)]
        x = F.gelu(self.input_proj(self.input_norm(feats))).transpose(1, 2) * masks[0]
        skips = [self.encoder[0](x, masks[0])]
        for level in range(1, 4):
            pooled = F.avg_pool1d(skips[-1], 2, stride=2, ceil_mode=True)
            x = self.down_proj[level - 1](pooled)
            masks.append(self._resize_mask(masks[0], x.shape[-1]))
            skips.append(self.encoder[level](x * masks[level], masks[level]))

        boundary: list[torch.Tensor | None] = [None, None, None, None]
        boundary[3] = self.boundary_bottom(skips[3], masks[3])
        for merge_index, level in enumerate((2, 1, 0)):
            up = self._upsample(boundary[level + 1], skips[level].shape[-1])
            boundary[level] = self.boundary_merge[merge_index](
                torch.cat([skips[level], up], dim=1), masks[level]
            )
        # All entries have been populated by the decoder above.
        boundary_features = [x for x in boundary if x is not None]
        boundary_logits = [
            self.boundary_out[i](boundary_features[i]) for i in range(4)
        ]

        state = self.class_bottom(skips[3], masks[3])
        state = self._condition(state, boundary_features[3], boundary_logits[3], masks[3], 3,
                                boundary_scale)
        for merge_index, level in enumerate((2, 1, 0)):
            up = self._upsample(state, skips[level].shape[-1])
            state = self.class_merge[merge_index](
                torch.cat([skips[level], up], dim=1), masks[level]
            )
            state = self._condition(state, boundary_features[level], boundary_logits[level],
                                    masks[level], level, boundary_scale)

        fine_prob = float(boundary_scale) * torch.sigmoid(boundary_logits[0])
        fine_boundary = float(boundary_scale) * boundary_features[0]
        if self.probability_gate:
            fine_boundary = fine_boundary * torch.sigmoid(boundary_logits[0])
        logits = self.class_out(torch.cat([state, fine_boundary, fine_prob], dim=1))
        return logits.transpose(1, 2) * valid.unsqueeze(-1), [
            x.squeeze(1) for x in boundary_logits
        ]


class BoundaryUNetLocalizer(nn.Module):
    """Standalone sparse-layer temporal U-Net; no BiLSTM, anchor, or seed parent."""

    RF = 400

    def __init__(self, name: str = "facebook/wav2vec2-xls-r-300m",
                 revision: str | None = None, layer: int = 8,
                 freeze: bool = True, n_layers_keep: int | None = 8,
                 hidden: int = 128, dropout: float = 0.1,
                 layerdrop: float = 0.0, wav_norm: bool = True,
                 probability_gate: bool = False,
                 symmetric_boundary: bool = False):
        super().__init__()
        if not freeze:
            raise ValueError("BoundaryUNetLocalizer requires a frozen SSL encoder")
        if layer is None:
            raise ValueError("BoundaryUNetLocalizer reads one explicit SSL layer")
        if n_layers_keep is None or n_layers_keep < layer:
            raise ValueError("n_layers_keep must include the selected SSL layer")
        self.frontend_name = "xlsr"
        self.wav_norm = wav_norm
        self.frontend = XLSRFrontend(name, revision=revision, layer=layer, freeze=True,
                                     n_layers_keep=n_layers_keep, layerdrop=layerdrop)
        self.symmetric_boundary = symmetric_boundary
        self.reader = BoundaryUNetReader(
            self.frontend.out_dim, hidden, dropout, probability_gate=probability_gate
        )
        self._aux: dict[str, object] = {}

    def param_groups(self, lr_ssl: float, lr_head: float):
        del lr_ssl
        return [{"params": list(self.reader.parameters()), "lr": lr_head,
                 "name": "head"}]

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor,
                n_segments: torch.Tensor, boundary_scale: float = 1.0) -> torch.Tensor:
        seg_max = int(n_segments.max())
        need = seg_max * HOP + self.RF
        if wav.shape[1] < need:
            wav = F.pad(wav, (0, need - wav.shape[1]))
        lengths = lengths.to(wav.device)
        if self.wav_norm:
            keep = (torch.arange(wav.shape[1], device=wav.device)[None, :]
                    < lengths[:, None]).to(wav.dtype)
            count = lengths.clamp(min=1).unsqueeze(1).to(wav.dtype)
            mean = (wav * keep).sum(1, keepdim=True) / count
            var = (((wav - mean) * keep) ** 2).sum(1, keepdim=True) / count
            wav = (wav - mean) * torch.rsqrt(var + 1e-7) * keep
        feats = self.frontend(wav, lengths)
        if feats.shape[1] < seg_max:
            feats = F.pad(feats, (0, 0, 0, seg_max - feats.shape[1]))
        feats = feats[:, :seg_max]
        valid = (torch.arange(seg_max, device=wav.device)[None, :]
                 < n_segments.to(wav.device)[:, None])
        logits, boundary_logits = self.reader(feats, valid, boundary_scale)
        self._aux = {"boundary_logits": boundary_logits, "valid": valid}
        return logits

    def boundary_loss(self, lab: torch.Tensor) -> torch.Tensor:
        """Balanced BCE on the exact transition cell at all four decoder scales."""
        logits = self._aux.get("boundary_logits")
        if not isinstance(logits, list):
            raise RuntimeError("call forward before boundary_loss")
        valid = lab != -100
        target = torch.zeros_like(lab, dtype=torch.float32)
        transition = (
            valid[:, 1:] & valid[:, :-1] & (lab[:, 1:] != lab[:, :-1])
        ).float()
        target[:, 1:] = transition
        if self.symmetric_boundary:
            target[:, :-1] = torch.maximum(target[:, :-1], transition)
        weighted_losses = []
        weights = (1.0, 0.5, 0.25, 0.125)
        for weight, scale_logits in zip(weights, logits):
            size = scale_logits.shape[1]
            scale_target = F.adaptive_max_pool1d(target[:, None], size)[:, 0]
            scale_valid = F.adaptive_max_pool1d(valid.float()[:, None], size)[:, 0] > 0
            y = scale_target[scale_valid]
            z = scale_logits.float()[scale_valid]
            positive = y.sum()
            pos_weight = ((y.numel() - positive) / positive.clamp_min(1.0)).clamp(max=100.0)
            weighted_losses.append(weight * F.binary_cross_entropy_with_logits(
                z, y, pos_weight=pos_weight
            ))
        self._aux["boundary_target"] = target.detach()
        return sum(weighted_losses) / sum(weights)

    @torch.inference_mode()
    def score(self, wav: torch.Tensor, lengths: torch.Tensor,
              n_segments: torch.Tensor) -> torch.Tensor:
        out = self(wav, lengths, n_segments)
        return out[..., 1] - out[..., 0]


def _log_matrix_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication in the log semiring."""
    return torch.logsumexp(left.unsqueeze(-1) + right.unsqueeze(-3), dim=-2)


def _inclusive_log_prefix(matrices: torch.Tensor) -> torch.Tensor:
    """All left-to-right inclusive matrix products in O(log S) GPU launches."""
    result = matrices
    offset = 1
    while offset < matrices.shape[1]:
        updated = result.clone()
        updated[:, offset:] = _log_matrix_product(
            result[:, :-offset], result[:, offset:]
        )
        result = updated
        offset *= 2
    return result


def _inclusive_log_suffix(matrices: torch.Tensor) -> torch.Tensor:
    """All left-to-right suffix matrix products in O(log S) GPU launches."""
    result = matrices
    offset = 1
    while offset < matrices.shape[1]:
        updated = result.clone()
        updated[:, :-offset] = _log_matrix_product(
            result[:, :-offset], result[:, offset:]
        )
        result = updated
        offset *= 2
    return result


def linear_chain_log_marginals(
    unary: torch.Tensor, transitions: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact log marginals and log partition for a padded first-order chain."""
    if unary.ndim != 3:
        raise ValueError(f"unary must have shape (B,S,C), got {tuple(unary.shape)}")
    batch, steps, classes = unary.shape
    expected = (batch, max(steps - 1, 0), classes, classes)
    if tuple(transitions.shape) != expected:
        raise ValueError(
            f"transitions must have shape {expected}, got {tuple(transitions.shape)}"
        )
    lengths = lengths.to(device=unary.device, dtype=torch.long)
    if tuple(lengths.shape) != (batch,):
        raise ValueError(f"lengths must have shape ({batch},), got {tuple(lengths.shape)}")
    if steps == 0 or bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError("every chain length must be in [1, S]")

    unary32 = unary.float()
    transition32 = transitions.float()
    if steps == 1:
        alpha_all = unary32
        beta_all = torch.zeros_like(unary32)
    else:
        # Link M_t(i,j) includes the destination unary u_(t+1)(j). Padded
        # links are log-semiring identities, so the scan is exact for ragged
        # batches without reading padded unary values.
        matrices = transition32 + unary32[:, 1:].unsqueeze(2)
        valid_link = torch.arange(steps - 1, device=unary.device)[None, :] < (
            lengths - 1
        )[:, None]
        diagonal = torch.eye(classes, dtype=torch.bool, device=unary.device)
        identity = unary32.new_full((classes, classes), float("-inf"))
        identity = identity.masked_fill(diagonal, 0.0)
        matrices = torch.where(
            valid_link[:, :, None, None], matrices, identity[None, None]
        )

        prefix = _inclusive_log_prefix(matrices)
        alpha_rest = torch.logsumexp(
            unary32[:, :1, :, None] + prefix, dim=2
        )
        alpha_all = torch.cat([unary32[:, :1], alpha_rest], dim=1)

        suffix = _inclusive_log_suffix(matrices)
        beta_before_last = torch.logsumexp(suffix, dim=3)
        beta_all = torch.cat(
            [beta_before_last, torch.zeros_like(unary32[:, :1])], dim=1
        )

    log_partition = torch.logsumexp(alpha_all[:, -1], dim=1)
    log_marginals = alpha_all + beta_all - log_partition[:, None, None]
    valid = torch.arange(steps, device=unary.device)[None, :] < lengths[:, None]
    log_marginals = log_marginals.masked_fill(~valid.unsqueeze(2), 0.0)
    return log_marginals, log_partition


def linear_chain_nll(
    unary: torch.Tensor,
    transitions: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    log_partition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-valid-cell negative log likelihood for a padded chain batch."""
    if log_partition is None:
        _, log_partition = linear_chain_log_marginals(unary, transitions, lengths)
    batch, steps, _ = unary.shape
    labels = labels.to(device=unary.device, dtype=torch.long)
    if tuple(labels.shape) != (batch, steps):
        raise ValueError(f"labels must have shape {(batch, steps)}, got {tuple(labels.shape)}")
    valid = torch.arange(steps, device=unary.device)[None, :] < lengths[:, None]
    safe_labels = labels.masked_fill(~valid, 0)
    unary_gold = unary.float().gather(2, safe_labels.unsqueeze(2)).squeeze(2)
    gold_score = (unary_gold * valid).sum(1)
    if steps > 1:
        previous = safe_labels[:, :-1]
        current = safe_labels[:, 1:]
        selected = transitions.float().gather(
            2,
            previous[:, :, None, None].expand(-1, -1, 1, transitions.shape[3]),
        ).squeeze(2)
        selected = selected.gather(2, current.unsqueeze(2)).squeeze(2)
        valid_link = torch.arange(steps - 1, device=unary.device)[None, :] < (
            lengths - 1
        )[:, None]
        gold_score = gold_score + (selected * valid_link).sum(1)
    return (log_partition - gold_score).sum() / lengths.float().sum()


def native_block_features(
    states: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize exact utterance-aligned blocks for a native coarse head.

    Returns ``(features, base_min, block_valid)``.  The feature at each block
    concatenates the second-pass state at the parent's minimum, the mean state,
    and seven score/count statistics. Invalid batch padding cannot enter any
    statistic, and the last partial utterance block remains a real block.
    """
    if states.ndim != 3 or scores.shape != states.shape[:2]:
        raise ValueError(
            f"expected states (B,S,D) and scores (B,S), got "
            f"{tuple(states.shape)} and {tuple(scores.shape)}"
        )
    if valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor matching scores")
    if factor < 2:
        raise ValueError("native block factor must be at least two")

    batch, steps, state_dim = states.shape
    pad = (-steps) % factor
    states32 = states.float()
    scores32 = scores.float()
    if pad:
        states32 = F.pad(states32, (0, 0, 0, pad))
        scores32 = F.pad(scores32, (0, pad))
        valid = F.pad(valid, (0, pad), value=False)
    blocks = states32.shape[1] // factor
    block_states = states32.view(batch, blocks, factor, state_dim)
    block_scores = scores32.view(batch, blocks, factor)
    block_mask = valid.view(batch, blocks, factor)
    block_valid = block_mask.any(dim=-1)
    count = block_mask.sum(dim=-1).clamp(min=1)

    high = torch.finfo(scores32.dtype).max
    low = torch.finfo(scores32.dtype).min
    for_min = block_scores.masked_fill(~block_mask, high)
    base_min, min_index = for_min.min(dim=-1)
    gather_index = min_index[..., None, None].expand(-1, -1, 1, state_dim)
    worst_state = block_states.gather(2, gather_index).squeeze(2)

    mask_float = block_mask.unsqueeze(-1).to(block_states.dtype)
    mean_state = (block_states * mask_float).sum(dim=2) / count[..., None]
    mean_score = (
        block_scores.masked_fill(~block_mask, 0.0).sum(dim=-1) / count
    )
    centred = (block_scores - mean_score[..., None]).masked_fill(~block_mask, 0.0)
    std_score = torch.sqrt(
        centred.square().sum(dim=-1) / count + 1e-6
    )
    max_score = block_scores.masked_fill(~block_mask, low).max(dim=-1).values
    two_lowest = for_min.topk(k=2, dim=-1, largest=False).values
    second_min = torch.where(count > 1, two_lowest[..., 1], base_min)
    fraction = count.to(scores32.dtype) / float(factor)
    min_position = min_index.to(scores32.dtype) / float(factor - 1)
    summaries = torch.stack(
        [base_min, second_min, mean_score, std_score, max_score,
         fraction, min_position],
        dim=-1,
    )
    features = torch.cat([worst_state, mean_state, summaries], dim=-1)
    features = features.masked_fill(~block_valid[..., None], 0.0)
    base_min = base_min.masked_fill(~block_valid, 0.0)
    return features, base_min, block_valid


class NativeBlockHead(nn.Module):
    """Residual native-grid predictor initialized to the parent's hard minimum."""

    SUMMARY_DIM = 7

    def __init__(self, state_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_norm = nn.LayerNorm(2 * self.state_dim)
        self.state_proj = nn.Linear(2 * self.state_dim, hidden)
        self.summary_proj = nn.Linear(self.SUMMARY_DIM, hidden // 2)
        self.delta = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden + hidden // 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self, features: torch.Tensor, base_min: torch.Tensor,
        block_valid: torch.Tensor,
    ) -> torch.Tensor:
        expected = 2 * self.state_dim + self.SUMMARY_DIM
        if features.shape[-1] != expected:
            raise ValueError(
                f"native features have width {features.shape[-1]}, expected {expected}"
            )
        state = self.state_norm(features[..., : 2 * self.state_dim])
        state = F.gelu(self.state_proj(state))
        summary = F.gelu(self.summary_proj(features[..., 2 * self.state_dim :]))
        correction = self.delta(torch.cat([state, summary], dim=-1)).squeeze(-1)
        score = base_min + correction
        return score.masked_fill(~block_valid, 0.0)


def build_model(blob: dict) -> nn.Module:
    """Instantiate the architecture recorded in a checkpoint blob (no weights loaded)."""
    arch = blob.get("arch", "localizer")
    if arch == "boundary_unet_localizer":
        return BoundaryUNetLocalizer(**blob.get("model_args", {}))
    if arch == "position_anchor_localizer":
        return PositionAnchorLocalizer(**blob.get("model_args", {}))
    if arch == "native_block_localizer":
        return NativeBlockLocalizer(**blob.get("model_args", {}))
    if arch == "dedicated_native160_localizer":
        return DedicatedNative160Localizer(**blob.get("model_args", {}))
    if arch == "native_pooled_localizer":
        return NativePooledLocalizer(**blob.get("model_args", {}))
    if arch == "native_event_localizer":
        from spanmark.models.event import NativeEventLocalizer
        return NativeEventLocalizer(**blob.get("model_args", {}))
    if arch == "native_red_localizer":
        from spanmark.models.red import NativeREDLocalizer
        return NativeREDLocalizer(**blob.get("model_args", {}))
    if arch == "native_segmental_localizer":
        from spanmark.models.segmental import NativeSegmentalLocalizer
        return NativeSegmentalLocalizer(**blob.get("model_args", {}))
    if arch == "native_coverage_localizer":
        from spanmark.models.coverage import NativeCoverageLocalizer
        return NativeCoverageLocalizer(**blob.get("model_args", {}))
    if arch == "native_frame_duration_localizer":
        from spanmark.models.frame_duration import NativeFrameDurationLocalizer
        return NativeFrameDurationLocalizer(**blob.get("model_args", {}))
    if arch == "native_interval_localizer":
        from spanmark.models.interval import NativeIntervalLocalizer
        return NativeIntervalLocalizer(**blob.get("model_args", {}))
    if arch == "native_witness_localizer":
        from spanmark.models.witness import NativeWitnessLocalizer
        return NativeWitnessLocalizer(**blob.get("model_args", {}))
    if arch == "boundary_crf_localizer":
        return BoundaryCRFLocalizer(**blob.get("model_args", {}))
    if arch == "anchor_localizer":
        return AnchorLocalizer(**blob.get("model_args", {}))
    if arch == "xlsr_localizer":
        return XLSRLocalizer(**blob.get("model_args", {}))
    return Localizer(blob.get("frontend", "mfcc"),
                     hidden=blob.get("hidden", 256), dropout=blob.get("dropout", 0.2))


class AnchorLocalizer(XLSRLocalizer):
    """Anchor-relative read-out: score every segment RELATIVE to the utterance's own
    most-genuine-looking segments.

    Both corpora are built by inserting synthetic spans into genuine recordings, so
    every utterance contains genuine speech. Pass 1 (frame head) gives P(bonafide);
    an attention over frames with weights softmax(tau * logit_1) selects the
    confidently-genuine frames and pools their BiLSTM states into an anchor a.
    Pass 2 scores each frame from [y_t ; y_t - a ; |y_t - a| ; cos(y_t, a) ; p1_t]
    through a small BiLSTM. A whole-utterance offset (channel, level, speaker,
    recording domain) shifts y_t and a together and cancels in y_t - a, which is the
    failure mode the label-free LPS probe identified. Loss = CE(pass 2) + 0.5 CE(pass 1).
    """

    def __init__(self, *a, anchor_tau: float = 1.0, anchor_mode: str = "soft", anchor_topk: float = 0.25,
                 anchor_source: str = "state", **kw):
        """anchor_mode: 'soft' = softmax(tau * logit) over all frames; 'topk' = uniform average of the
        top `anchor_topk` fraction of frames by pass-1 bonafide logit (hard, straight-through-free:
        the anchor is a constant w.r.t. the selection, gradients flow through the selected states).
        anchor_source: 'state' = anchor built from BiLSTM states y; 'feats' = also from the projected
        frozen features x (before the BiLSTM), concatenated -- a representation less shaped by PS labels."""
        super().__init__(*a, **kw)
        d = self.out.in_features
        h = self.proj[0].out_features
        self.anchor_mode, self.anchor_topk, self.anchor_source = anchor_mode, anchor_topk, anchor_source
        self.log_tau = nn.Parameter(torch.tensor(float(anchor_tau)).log())
        extra_feat = (3 * h + 1) if anchor_source == "feats" else 0
        self.anchor_in = nn.Sequential(nn.Linear(3 * d + 2 + extra_feat, h), nn.GELU())
        self.rnn2 = nn.LSTM(h, h, num_layers=1, batch_first=True, bidirectional=True)
        self.out2 = nn.Linear(2 * h, 2)
        self._aux = {}

    def _head_forward(self, feats, valid, n_segments, seg_max):
        if self.feat_norm == "none":
            x = self.norm(feats)
        else:
            v = valid.unsqueeze(-1).to(feats.dtype)
            cnt = v.sum(1, keepdim=True).clamp(min=1.0)
            mu = (feats * v).sum(1, keepdim=True) / cnt
            centred = feats - mu
            if self.feat_norm == "mean":
                x = self.norm(centred)
            elif self.feat_norm == "meanstd":
                var = ((centred * v) ** 2).sum(1, keepdim=True) / cnt
                x = self.norm(centred * torch.rsqrt(var + 1e-5))
            else:
                x = torch.cat([self.norm(feats), self.norm2(centred)], dim=-1)
        x = self.proj(x) * valid.unsqueeze(-1)
        lens = n_segments.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=seg_max)
        y = self.drop(y)
        logits1 = self.out(y)                                                  # (B, S, 2)
        # ---- anchor: attention over confidently-genuine frames of the SAME utterance ----
        d1 = (logits1[..., 1] - logits1[..., 0]).float()
        if self.anchor_mode == "topk":
            # top fraction of VALID frames by pass-1 bonafide logit, uniform weights
            n_valid = valid.sum(1, keepdim=True).float()
            k = (n_valid * self.anchor_topk).ceil().clamp(min=1.0)                     # (B, 1)
            ranks = d1.masked_fill(~valid, float("-inf")).argsort(dim=1, descending=True).argsort(dim=1)
            sel = (ranks.float() < k) & valid
            w = (sel.float() / sel.float().sum(1, keepdim=True).clamp(min=1.0)).to(y.dtype)
        else:
            att = (d1 * self.log_tau.exp()).masked_fill(~valid, float("-inf"))
            w = torch.softmax(att, dim=1).to(y.dtype)                              # (B, S)
        anchor = torch.einsum("bs,bsd->bd", w, y)                              # (B, D)
        a = anchor.unsqueeze(1).expand_as(y)
        cos = F.cosine_similarity(y.float(), a.float(), dim=-1).unsqueeze(-1).to(y.dtype)
        p1 = torch.softmax(logits1.float(), dim=-1)[..., 1:2].to(y.dtype)
        parts = [y, y - a, (y - a).abs(), cos, p1]
        if self.anchor_source == "feats":
            ax = torch.einsum("bs,bsd->bd", w, x).unsqueeze(1).expand_as(x)  # anchor in projected-feature space
            cosx = F.cosine_similarity(x.float(), ax.float(), dim=-1).unsqueeze(-1).to(y.dtype)
            parts += [x, x - ax, (x - ax).abs(), cosx]
        z = self.anchor_in(torch.cat(parts, dim=-1)) * valid.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
        y2, _ = self.rnn2(packed)
        y2, _ = nn.utils.rnn.pad_packed_sequence(y2, batch_first=True, total_length=seg_max)
        self._aux = {"logits1": logits1, "anchor_w": w, "y2": y2}
        return self.out2(self.drop(y2))

    def aux_loss(self, lab: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self._aux["logits1"].float().reshape(-1, 2), lab.reshape(-1), ignore_index=-100)


class PositionAnchorLocalizer(AnchorLocalizer):
    """Anchor reader trained with class-by-position supervision.

    The auxiliary task follows SAL: every valid frame is Start, Middle, End,
    or Unit within its contiguous bonafide/spoof run, yielding eight classes.
    The auxiliary logits shape the shared anchor states during training but are
    not fused into the inference score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_out = nn.Linear(self.out2.in_features, 8)

    @staticmethod
    def position_targets(lab: torch.Tensor) -> torch.Tensor:
        """Map binary frame labels to {fake,real} x {S,M,E,U}; keep padding ignored."""
        target = torch.full_like(lab, -100)
        for batch_index in range(lab.shape[0]):
            n = int((lab[batch_index] != -100).sum())
            if n == 0:
                continue
            labels = lab[batch_index, :n]
            start = torch.ones(n, dtype=torch.bool, device=lab.device)
            end = torch.ones(n, dtype=torch.bool, device=lab.device)
            if n > 1:
                change = labels[1:] != labels[:-1]
                start[1:] = change
                end[:-1] = change
            position = torch.ones(n, dtype=torch.long, device=lab.device)  # Middle
            position[start] = 0                                           # Start
            position[end] = 2                                             # End
            position[start & end] = 3                                     # Unit
            target[batch_index, :n] = 4 * labels + position
        return target

    def position_loss(self, lab: torch.Tensor) -> torch.Tensor:
        states = self._aux.get("y2")
        if not isinstance(states, torch.Tensor):
            raise RuntimeError("call forward before position_loss")
        # Training calls auxiliary losses after leaving the forward autocast block;
        # restore the head's parameter dtype explicitly instead of mixing FP16 states
        # with FP32 weights.
        logits = self.position_out(states.to(self.position_out.weight.dtype))
        target = self.position_targets(lab)
        return F.cross_entropy(
            logits.float().reshape(-1, 8), target.reshape(-1), ignore_index=-100
        )


class BAMSelfWeightedPooling(nn.Module):
    """The official BAM mean-only attentive pool over aligned SSL frames.

    BAM's ``SelfWeightedPooling`` owns one ``D x 1`` weight matrix, applies
    ``softmax(tanh(x @ w))`` over the frames of a resolution block, and emits
    their weighted mean.  This implementation additionally carries a validity
    mask so an utterance's final partial block is legal and batch padding can
    never affect its representation.  The resolution-specific reader and
    fresh classifier deliberately live after this module, matching BAM's
    released ``SSL -> pool -> coarse reader -> FC`` ordering.
    """

    def __init__(self, state_dim: int, block_size: int = 8):
        super().__init__()
        if state_dim < 1 or block_size < 2:
            raise ValueError("BAM native pooling dimensions must be positive")
        self.state_dim = int(state_dim)
        self.block_size = int(block_size)
        self.weight = nn.Parameter(torch.empty(self.state_dim, 1))
        # BAM's released SelfWeightedPooling calls kaiming_uniform_ without
        # overriding its gain/nonlinearity arguments.  Preserve that exact
        # initialization rather than borrowing nn.Linear's a=sqrt(5) reset.
        nn.init.kaiming_uniform_(self.weight)

    @staticmethod
    def _blockify(x: torch.Tensor, block_size: int, pad_value: float) -> torch.Tensor:
        pad = (-x.shape[1]) % block_size
        if x.ndim == 2:
            return F.pad(x, (0, pad), value=pad_value).view(
                x.shape[0], -1, block_size
            )
        if x.ndim == 3:
            return F.pad(x, (0, 0, 0, pad), value=pad_value).view(
                x.shape[0], -1, block_size, x.shape[-1]
            )
        raise ValueError(f"expected rank-2 or rank-3 sequence, got rank {x.ndim}")

    def forward(
        self,
        states: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(block_states, block_valid, attention_weights)``."""
        if states.ndim != 3 or valid.shape != states.shape[:2]:
            raise ValueError("BAM native pool expects states (B,T,D) and valid (B,T)")
        if states.shape[-1] != self.state_dim or valid.dtype != torch.bool:
            raise ValueError("BAM native pool state width or validity dtype is wrong")

        blocks = self._blockify(states.float(), self.block_size, 0.0)
        block_mask = self._blockify(valid, self.block_size, 0.0).bool()
        block_valid = block_mask.any(dim=-1)
        attention_logits = torch.tanh(torch.matmul(blocks, self.weight).squeeze(-1))
        weights = torch.softmax(
            attention_logits.masked_fill(~block_mask, -1e4), dim=-1
        ) * block_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = (weights.unsqueeze(-1) * blocks).sum(dim=2)
        pooled = pooled.masked_fill(~block_valid.unsqueeze(-1), 0.0)
        return pooled, block_valid, weights


class DedicatedNative160Localizer(PositionAnchorLocalizer):
    """Seed-initialized model whose complete coarse graph predicts 160 ms.

    Consecutive SSL states are attentively pooled before the inherited
    pass-one/anchor/pass-two reader, and ``out2`` is a fresh resolution-specific
    classifier.  This is a coarse model rather than a branch attached to the
    seed's final frame states.  Training consumes only ``native_logits``;
    deployment uses a separate immutable seed model for the mandatory 20 ms
    stream.
    """

    dedicated_native_resolution_ms = 160

    def __init__(self, *args, native_block_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        if int(native_block_size) != 8:
            raise ValueError("the dedicated native-160 model requires eight 20 ms frames")
        self.native_block_size = int(native_block_size)
        self.native_pool = BAMSelfWeightedPooling(
            self.frontend.out_dim, block_size=self.native_block_size
        )
        # The fine classifier is replaced, not warm-started.  Pooling occurs
        # before the complete coarse reader, so this fresh FC consumes the
        # reader's coarse second-pass states exactly where the seed's old FC
        # consumed its fine states.
        self.out2 = nn.Linear(self.out2.in_features, 2)
        # Position supervision is absent from the exclusive native objective,
        # and SpecAugment is disabled inside XLSRFrontend.  Neither tensor is
        # loss-reachable; all other reader and SSL tensors remain eligible for
        # end-to-end fine-tuning under ``freeze=False``.
        self.position_out.requires_grad_(False)
        self.frontend.model.masked_spec_embed.requires_grad_(False)

    def _head_forward(self, feats, valid, n_segments, seg_max):
        # BAM pools the SSL sequence *before* its boundary/authenticity reader.
        # Keep the official attention arithmetic in FP32 under the outer XLS-R
        # autocast context; gradients still flow into every fine SSL state.
        with torch.autocast(device_type=feats.device.type, enabled=False):
            coarse_feats, native_valid, native_attention = self.native_pool(
                feats, valid
            )
        coarse_segments = torch.div(
            n_segments + self.native_block_size - 1,
            self.native_block_size,
            rounding_mode="floor",
        )
        coarse_max = coarse_feats.shape[1]
        expected_valid = (
            torch.arange(coarse_max, device=feats.device)[None, :]
            < coarse_segments.to(feats.device)[:, None]
        )
        if not torch.equal(native_valid, expected_valid):
            raise RuntimeError("native pool validity disagrees with ceil(n20/8)")
        native_logits = super()._head_forward(
            coarse_feats, native_valid, coarse_segments, coarse_max
        )
        self._aux["native_logits"] = native_logits
        self._aux["native_attention"] = native_attention
        self._aux["native_scores"] = {
            160: (native_logits[..., 1] - native_logits[..., 0]).float()
        }
        self._aux["native_valid"] = {160: native_valid}
        return native_logits

    @torch.inference_mode()
    def score_with_native(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor,
        n_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        native_logits = self(wav, lengths, n_segments)
        native_score = native_logits[..., 1] - native_logits[..., 0]
        native = self._aux.get("native_scores")
        if not isinstance(native, dict) or set(native) != {160}:
            raise RuntimeError("dedicated model did not produce only native 160 ms scores")
        return native_score, native


class NativeBlockLocalizer(PositionAnchorLocalizer):
    """Frozen anchor reader plus trained native-resolution block heads."""

    def __init__(
        self, *args, native_resolutions: tuple[int, ...] = (640,),
        native_hidden: int = 128, native_dropout: float = 0.1, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        resolutions = tuple(sorted({int(ms) for ms in native_resolutions}))
        if not resolutions or any(ms not in (40, 80, 160, 320, 640)
                                  for ms in resolutions):
            raise ValueError("native_resolutions must be drawn from 40..640 ms")
        self.native_resolutions = resolutions
        state_dim = self.out2.in_features
        self.native_heads = nn.ModuleDict({
            str(ms): NativeBlockHead(state_dim, native_hidden, native_dropout)
            for ms in resolutions
        })

    def _head_forward(self, feats, valid, n_segments, seg_max):
        base_logits = super()._head_forward(feats, valid, n_segments, seg_max)
        states = self._aux["y2"]
        scores = (base_logits[..., 1] - base_logits[..., 0]).float()
        native_scores: dict[int, torch.Tensor] = {}
        native_valid: dict[int, torch.Tensor] = {}
        # The cache trainer consumes FP32 features and trains the small heads in
        # FP32. Graded XLS-R inference runs under an outer FP16 autocast context;
        # disable it here so cached-development and live head rankings coincide.
        with torch.autocast(device_type=states.device.type, enabled=False):
            for ms in self.native_resolutions:
                factor = ms // 20
                features, base_min, block_valid = native_block_features(
                    states, scores, valid, factor
                )
                native_scores[ms] = self.native_heads[str(ms)](
                    features, base_min, block_valid
                )
                native_valid[ms] = block_valid
        self._aux["native_scores"] = native_scores
        self._aux["native_valid"] = native_valid
        return base_logits

    @torch.inference_mode()
    def score_with_native(
        self, wav: torch.Tensor, lengths: torch.Tensor,
        n_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        logits = self(wav, lengths, n_segments)
        fine = logits[..., 1] - logits[..., 0]
        native = self._aux.get("native_scores")
        if not isinstance(native, dict):
            raise RuntimeError("native heads did not produce scores")
        return fine, native


class CoarseTemporalResidualBlock(nn.Module):
    """Zero-started temporal residual used only by native coarse predictions.

    A width-five depthwise convolution keeps the parameter count small while
    dilation expands context.  The final projection starts at exact zero, so
    adding any number of these blocks is initially an identity transform.
    """

    def __init__(
        self,
        state_dim: int,
        bottleneck: int = 64,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if state_dim < 1 or bottleneck < 1 or dilation < 1:
            raise ValueError("coarse trunk dimensions and dilation must be positive")
        self.norm = nn.LayerNorm(state_dim)
        self.down = nn.Linear(state_dim, bottleneck)
        self.depthwise = nn.Conv1d(
            bottleneck,
            bottleneck,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=bottleneck,
        )
        self.mix = nn.Conv1d(bottleneck, bottleneck, kernel_size=1)
        self.up = nn.Linear(bottleneck, state_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or valid.shape != states.shape[:2]:
            raise ValueError("coarse residual block expects states (B,T,D) and valid (B,T)")
        mask = valid.unsqueeze(-1).to(states.dtype)
        hidden = F.gelu(self.down(self.norm(states))) * mask
        hidden = self.depthwise(hidden.transpose(1, 2))
        hidden = F.gelu(self.mix(F.gelu(hidden))).transpose(1, 2) * mask
        return (states + self.drop(self.up(hidden))) * mask


class NativeSummaryNorm(nn.LayerNorm):
    """Retain the fitted mean/std norm while separately normalizing new inputs."""

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        original_width = self.normalized_shape[0]
        base = super().forward(summary[..., :original_width])
        extra = summary[..., original_width:]
        if not extra.shape[-1]:
            return base
        extra = F.layer_norm(extra, (extra.shape[-1],), eps=self.eps)
        return torch.cat([base, extra], dim=-1)


class NativePooledClassifier(nn.Module):
    """Untethered attentive-statistics classifiers on native coarse grids.

    Unlike ``NativeBlockHead`` and the first team attentive head, no scalar
    fine score is carried into the native output.  Each resolution pools
    acoustic states first and classifies the resulting weighted mean/std.
    Optional temporal residual blocks form a shared coarse-only trunk before
    the resolution-specific pools.
    """

    _DILATIONS = (1, 2, 4)

    def __init__(
        self,
        state_dim: int,
        resolutions_ms: tuple[int, ...] = (160, 320, 640),
        attention_hidden: int = 64,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        trunk_layers: int = 0,
        trunk_bottleneck: int = 64,
        zero_output_init: bool = True,
        summary_mode: str = "meanstd",
        block_layers: int = 0,
        block_bottleneck: int = 64,
    ):
        super().__init__()
        resolutions = tuple(int(ms) for ms in resolutions_ms)
        if not resolutions or len(set(resolutions)) != len(resolutions):
            raise ValueError("native resolutions must be nonempty and unique")
        if any(ms <= 20 or ms % 20 for ms in resolutions):
            raise ValueError("native resolutions must be multiples of 20 ms above 20")
        if attention_hidden < 1 or classifier_hidden < 1:
            raise ValueError("native attention and classifier widths must be positive")
        if trunk_layers < 0 or trunk_layers > len(self._DILATIONS):
            raise ValueError(f"trunk_layers must be between 0 and {len(self._DILATIONS)}")
        if block_layers < 0 or block_layers > len(self._DILATIONS):
            raise ValueError(f"block_layers must be between 0 and {len(self._DILATIONS)}")
        if summary_mode not in ("meanstd", "extrema", "variation"):
            raise ValueError("unknown native summary mode")

        self.state_dim = int(state_dim)
        self.summary_mode = summary_mode
        summary_groups = {"meanstd": 2, "extrema": 4, "variation": 5}[summary_mode]
        self.resolutions_ms = resolutions
        self.block_sizes = {ms: ms // 20 for ms in resolutions}
        self.attention = nn.ModuleDict()
        self.classifier = nn.ModuleDict()

        # Construct common arm-1 modules before optional trunk parameters.  A
        # fixed seed therefore initializes every shared tensor bit-identically
        # when trunk depth is the sole causal change in arm 2.
        for ms in resolutions:
            key = str(ms)
            self.attention[key] = nn.Sequential(
                nn.LayerNorm(self.state_dim),
                nn.Linear(self.state_dim, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1, bias=False),
            )
            branch = nn.Sequential(
                (nn.LayerNorm(2 * self.state_dim) if summary_mode == "meanstd"
                 else NativeSummaryNorm(2 * self.state_dim)),
                nn.Linear(summary_groups * self.state_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, 1),
            )
            if zero_output_init:
                # A constant-zero native score is an unbiased start for frozen
                # readout transfer.  A dedicated end-to-end resolution model
                # disables this: its fresh classifier must send a nonzero
                # gradient through the attentive pool and base on update one.
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
            self.classifier[key] = branch

        self.trunk = nn.ModuleList([
            CoarseTemporalResidualBlock(
                self.state_dim,
                bottleneck=trunk_bottleneck,
                dilation=self._DILATIONS[index],
                dropout=dropout,
            )
            for index in range(trunk_layers)
        ])
        # Learn local frame interactions before reducing them to statistics.
        # Each block has its own mask and padding; neighboring coarse decisions
        # cannot leak into an independently sampled training block.
        self.block_trunk = nn.ModuleList([
            CoarseTemporalResidualBlock(
                self.state_dim,
                bottleneck=block_bottleneck,
                dilation=self._DILATIONS[index],
                dropout=dropout,
            )
            for index in range(block_layers)
        ])

    @staticmethod
    def _blockify(x: torch.Tensor, block_size: int, pad_value: float) -> torch.Tensor:
        pad = (-x.shape[1]) % block_size
        if x.ndim == 2:
            return F.pad(x, (0, pad), value=pad_value).view(
                x.shape[0], -1, block_size
            )
        if x.ndim == 3:
            return F.pad(x, (0, 0, 0, pad), value=pad_value).view(
                x.shape[0], -1, block_size, x.shape[-1]
            )
        raise ValueError(f"expected rank-2 or rank-3 sequence, got rank {x.ndim}")

    def _refine_native_score(
        self, ms: int, block_states: torch.Tensor,
        block_mask: torch.Tensor, score: torch.Tensor,
    ) -> torch.Tensor:
        """Allow a trained native decision to use the post-local frame states."""
        return score

    def forward(
        self,
        states: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if states.ndim != 3 or valid.shape != states.shape[:2]:
            raise ValueError("native pooled classifier expects states (B,T,D) and valid (B,T)")
        if states.shape[-1] != self.state_dim or valid.dtype != torch.bool:
            raise ValueError("native pooled classifier state width or valid dtype is wrong")

        encoded = states.float() * valid.unsqueeze(-1)
        for block in self.trunk:
            encoded = block(encoded, valid)

        scores: dict[int, torch.Tensor] = {}
        masks: dict[int, torch.Tensor] = {}
        for ms in self.resolutions_ms:
            key = str(ms)
            factor = self.block_sizes[ms]
            block_states = self._blockify(encoded, factor, 0.0)
            block_mask = self._blockify(valid, factor, 0.0).bool()
            block_valid = block_mask.any(dim=-1)
            if self.block_trunk:
                block_shape = block_states.shape
                local = block_states.reshape(-1, factor, self.state_dim)
                local_mask = block_mask.reshape(-1, factor)
                for layer in self.block_trunk:
                    local = layer(local, local_mask)
                block_states = local.reshape(block_shape)

            attention_logits = self.attention[key](block_states).squeeze(-1)
            weights = torch.softmax(
                attention_logits.masked_fill(~block_mask, -1e4), dim=-1
            ) * block_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mean = (weights.unsqueeze(-1) * block_states).sum(dim=2)
            variance = (
                weights.unsqueeze(-1) * (block_states - mean.unsqueeze(2)).square()
            ).sum(dim=2)
            std = (variance.clamp_min(0.0) + 1e-6).sqrt()
            summary = torch.cat([mean, std], dim=-1)
            if self.summary_mode != "meanstd":
                minimum = block_states.masked_fill(
                    ~block_mask.unsqueeze(-1), float("inf")
                ).amin(dim=2).masked_fill(~block_valid.unsqueeze(-1), 0.0)
                maximum = block_states.masked_fill(
                    ~block_mask.unsqueeze(-1), -float("inf")
                ).amax(dim=2).masked_fill(~block_valid.unsqueeze(-1), 0.0)
                summary = torch.cat([summary, minimum, maximum], dim=-1)
                if self.summary_mode == "variation":
                    adjacent = block_mask[..., 1:] & block_mask[..., :-1]
                    delta = (block_states[..., 1:, :] - block_states[..., :-1, :]).abs()
                    variation = (delta * adjacent.unsqueeze(-1)).sum(dim=2)
                    variation = variation / adjacent.sum(dim=-1, keepdim=True).clamp_min(1)
                    summary = torch.cat([summary, variation], dim=-1)
            native = self.classifier[key](summary).squeeze(-1).float()
            native = self._refine_native_score(ms, block_states, block_mask, native)
            scores[ms] = native.masked_fill(~block_valid, 0.0)
            masks[ms] = block_valid
        return scores, masks


class NativePooledLocalizer(PositionAnchorLocalizer):
    """Anchor reader plus untethered state-pooled native classifiers.

    The reader may be frozen for cached-head transfer or trained end to end for
    a dedicated resolution model; deployment metadata, rather than this class,
    decides which model owns the mandatory 20 ms stream.
    """

    def __init__(
        self,
        *args,
        native_resolutions_ms: tuple[int, ...] = (160, 320, 640),
        native_attention_hidden: int = 64,
        native_classifier_hidden: int = 128,
        native_dropout: float = 0.1,
        native_trunk_layers: int = 0,
        native_trunk_bottleneck: int = 64,
        native_zero_output_init: bool = True,
        native_summary_mode: str = "meanstd",
        native_block_layers: int = 0,
        native_block_bottleneck: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.native_pool = NativePooledClassifier(
            self.out2.in_features,
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

    def _head_forward(self, feats, valid, n_segments, seg_max):
        base_logits = super()._head_forward(feats, valid, n_segments, seg_max)
        # The frozen cache and live inference both use FP32 native modules.
        # Disable the outer XLS-R autocast so their rankings match exactly.
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
        if not isinstance(native, dict):
            raise RuntimeError("native pooled classifier did not produce scores")
        return fine, native


class BoundaryCRFLocalizer(PositionAnchorLocalizer):
    """Exact focal unary reader plus an observation-conditioned label chain.

    The archived treatment keeps the inherited unary reader unchanged. A
    separate head sees detached adjacent second-pass states and learns four
    directed transition potentials through sequence likelihood. Inference
    returns forward-backward marginal log odds in one model forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.out2.in_features
        self.transition_out = nn.Linear(3 * state_dim, 4, bias=False)
        nn.init.zeros_(self.transition_out.weight)

    def _apply_structured_reader(
        self,
        base_logits: torch.Tensor,
        states: torch.Tensor,
        valid: torch.Tensor,
        n_segments: torch.Tensor,
    ) -> torch.Tensor:
        detached_states = states.detach()
        if detached_states.shape[1] > 1:
            left, right = detached_states[:, :-1], detached_states[:, 1:]
            link_features = torch.cat([left, right, (right - left).abs()], dim=-1)
            transition = self.transition_out(link_features).view(
                base_logits.shape[0], base_logits.shape[1] - 1, 2, 2
            )
        else:
            transition = base_logits.new_zeros(base_logits.shape[0], 0, 2, 2)
        detached_unary = base_logits.detach()
        lengths = n_segments.to(device=base_logits.device, dtype=torch.long)
        log_marginals, log_partition = linear_chain_log_marginals(
            detached_unary, transition, lengths
        )
        self._aux.update({
            "base_logits": base_logits,
            "crf_unary": detached_unary,
            "crf_transition": transition,
            "crf_log_partition": log_partition,
            "crf_valid": valid,
            "crf_lengths": lengths,
        })
        return log_marginals

    def _head_forward(self, feats, valid, n_segments, seg_max):
        base_logits = super()._head_forward(feats, valid, n_segments, seg_max)
        states = self._aux["y2"]
        return self._apply_structured_reader(
            base_logits, states, valid, n_segments
        )

    def structured_loss(self, labels: torch.Tensor) -> torch.Tensor:
        required = (
            "crf_unary", "crf_transition", "crf_log_partition", "crf_lengths"
        )
        if any(name not in self._aux for name in required):
            raise RuntimeError("call forward before structured_loss")
        return linear_chain_nll(
            self._aux["crf_unary"],
            self._aux["crf_transition"],
            labels,
            self._aux["crf_lengths"],
            self._aux["crf_log_partition"],
        )
