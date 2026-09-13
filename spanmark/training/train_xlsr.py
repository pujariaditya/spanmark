#!/usr/bin/env python
"""Fine-tune an XLS-R localizer on PartialSpoof train with random crops.

    python train_xlsr.py --out ckpt/my_model.pt --epochs 5

Per-segment CE on the 20 ms grid, crops taken at segment boundaries so labels
stay index-aligned. Selection on dev segment EER (the graded quantity).
Augmentation hooks (splice / noise) live in pslps/augment.py and are off unless
requested.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from spanmark.data import HOP, SAMPLE_RATE, SegmentDataset, collate, load_label_npz, read_manifest
from spanmark.models import (
    AnchorLocalizer,
    BoundaryUNetLocalizer,
    NativePooledLocalizer,
    PositionAnchorLocalizer,
    XLSRLocalizer,
)
from spanmark.train import seg_eer

TASKDATA = os.environ.get(
    "SPANMARK_TASKDATA",
    os.environ.get("PSLPS_TASKDATA",
                   str(pathlib.Path(__file__).resolve().parents[2] / "data" / "labels")))
# Manifests ship with paths relative to the corpus root. REMAP rewrites a
# manifest's recorded prefix onto wherever the corpus actually lives, so a
# manifest built on one machine still resolves on another.
REMAP = (os.environ.get("SPANMARK_MANIFEST_PREFIX", "PartialSpoof/database/"),
         os.environ.get("SPANMARK_CORPUS_ROOT", ""))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp16_tensor_fingerprint(
    tensors: dict[str, torch.Tensor] | list[tuple[str, torch.Tensor]],
) -> str:
    """Hash named tensors canonically after the checkpoint's FP16 projection."""
    items = tensors.items() if isinstance(tensors, dict) else tensors
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.half()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def remap(items):
    from spanmark.data import Utt
    out = []
    for u in items:
        p = u.path.replace(REMAP[0], REMAP[1]) if Path(REMAP[1]).exists() else u.path
        out.append(Utt(u.utt_id, u.n_segments, p))
    return out


class CropDataset(Dataset):
    """Random crop of `crop_segments` segments (aligned to the 20 ms grid)."""

    def __init__(self, items, labels, crop_segments=200, augment=None,
                 crop_alignment=1, admit_partial_tail=False,
                 return_crop_metadata=False):
        if crop_segments < 1 or crop_alignment < 1 or crop_segments % crop_alignment:
            raise ValueError("crop alignment must divide the crop length")
        self.items, self.labels, self.crop, self.augment = items, labels, crop_segments, augment
        self.crop_alignment = int(crop_alignment)
        self.admit_partial_tail = bool(admit_partial_tail)
        self.return_crop_metadata = bool(return_crop_metadata)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav, sr = sf.read(it.path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        y = np.asarray(self.labels[it.utt_id], dtype=np.int64)
        n = min(it.n_segments, len(y))
        if self.augment is not None:
            wav, y = self.augment(wav, y, it)
            n = min(len(y), math.ceil(len(wav) / HOP))
        source_n, start = n, 0
        source_labels = y
        if n > self.crop:
            # Coarse-native objectives must see the same utterance-origin phase
            # as inference.  An aligned start plus an alignment-divisible crop
            # contains only complete non-tail blocks.
            upper = (n - self.crop) // self.crop_alignment
            if self.admit_partial_tail and (n - self.crop) % self.crop_alignment:
                upper += 1
            s = random.randint(0, upper)
            s *= self.crop_alignment
            start = s
            y = y[s: s + self.crop]
            wav = wav[s * HOP: (s + self.crop) * HOP]
            n = min(self.crop, source_n - s)
        else:
            y = y[:n]
            wav = wav[: n * HOP]
        sample = (torch.from_numpy(np.ascontiguousarray(wav)), torch.from_numpy(y), n, it.utt_id)
        if not self.return_crop_metadata:
            return sample
        if sr != SAMPLE_RATE or len(y) != n or n <= 0 or not len(wav) <= n * HOP:
            raise RuntimeError("invalid audited crop waveform or label length")
        # Derive targets from the source grid independently of batch padding.
        targets = [int(source_labels[k:min(k + self.crop_alignment, source_n)].min())
                   for k in range(start, start + n, self.crop_alignment)]
        metadata = {
            "utt_id": it.utt_id, "source_n": source_n, "start": start,
            "n": n, "crop": self.crop, "alignment": self.crop_alignment,
            "admit_partial_tail": self.admit_partial_tail,
            "tail_admitted": source_n > self.crop and n < self.crop,
            "samples": len(wav), "native_targets": targets,
        }
        return (*sample, metadata)


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, S, Y = [], [], []
    native_scores: dict[int, list[np.ndarray]] = {}
    native_labels: dict[int, list[np.ndarray]] = {}
    for wav, lengths, lab, mask, nsegs, _ in loader:
        wav, lab, mask = wav.to(device), lab.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(wav, lengths, nsegs)
        out = out.float()
        losses.append(criterion(out.reshape(-1, 2), lab.reshape(-1)).item())
        s = (out[..., 1] - out[..., 0])[mask].cpu().numpy()
        S.append(s); Y.append(lab[mask].cpu().numpy())
        emitted = getattr(model, "_aux", {}).get("native_scores", {})
        emitted_valid = getattr(model, "_aux", {}).get("native_valid", {})
        if isinstance(emitted, dict) and isinstance(emitted_valid, dict):
            for resolution_ms, prediction in emitted.items():
                if resolution_ms not in emitted_valid or resolution_ms % 20:
                    raise RuntimeError(
                        f"invalid native evaluation output at {resolution_ms} ms"
                    )
                target, target_valid = native_block_targets(
                    lab, resolution_ms // 20
                )
                usable = emitted_valid[resolution_ms] & target_valid
                if prediction.shape != target.shape or usable.shape != target.shape:
                    raise RuntimeError(
                        "native evaluation score/target mismatch: "
                        f"scores={tuple(prediction.shape)}, target={tuple(target.shape)}"
                    )
                native_scores.setdefault(resolution_ms, []).append(
                    prediction[usable].float().cpu().numpy()
                )
                native_labels.setdefault(resolution_ms, []).append(
                    target[usable].long().cpu().numpy()
                )
    native_eer = {
        resolution_ms: seg_eer(
            np.concatenate(native_scores[resolution_ms]),
            np.concatenate(native_labels[resolution_ms]),
        )
        for resolution_ms in sorted(native_scores)
    }
    return (
        float(np.mean(losses)),
        seg_eer(np.concatenate(S), np.concatenate(Y)),
        native_eer,
    )


MULTIRES_BLOCKS = (2, 4, 8, 16, 32)
ANCHOR_SECOND_PASS_PREFIXES = ("anchor_in.", "rnn2.", "out2.")
ANCHOR_SECOND_PASS_PARAMETER_COUNT = 1_447_682
ANCHOR_PREPASS_PREFIXES = ("norm.", "norm2.", "proj.", "rnn.", "out.")
ANCHOR_PREPASS_EXACT_NAMES = ("log_tau",)
ANCHOR_PREPASS_PARAMETER_COUNT = 3_159_299
NATIVE_REPLACED_CLASSIFIER_KEYS = (
    "out2.weight", "out2.bias", "position_out.weight", "position_out.bias",
)


def resolve_schedule_epochs(training_epochs: int, schedule_epochs: int) -> int:
    """Return a validated cosine horizon independent of the loop duration."""
    if schedule_epochs < 0:
        raise ValueError("--schedule-epochs must be nonnegative")
    if schedule_epochs and schedule_epochs < training_epochs:
        raise ValueError("--schedule-epochs must be at least --epochs")
    return schedule_epochs or training_epochs


def compact_state_dict(model: nn.Module, frozen_base: bool) -> dict[str, torch.Tensor]:
    """Omit frozen SSL tensors while retaining every selectively tuned one."""
    trained_encoder = {
        name for name, parameter in model.named_parameters()
        if name.startswith("frontend.model.") and parameter.requires_grad
    }
    return {
        name: value
        for name, value in model.state_dict().items()
        if not (
            frozen_base
            and name.startswith("frontend.model.")
            and name not in trained_encoder
        )
    }


def load_checkpoint_blob(path: str | Path) -> dict:
    """Load either a regular training checkpoint or its gzip-packed form."""
    checkpoint = Path(path)
    if checkpoint.suffix == ".gz":
        with gzip.open(checkpoint, "rb") as stream:
            return torch.load(stream, map_location="cpu", weights_only=False)
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def initialization_state(
    blob: dict,
    reader_only: bool = False,
    model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Return fp32 initialization tensors, optionally excluding the SSL model.

    ``reader_only`` deliberately retains ``frontend.layer_weights``: the
    all-state mixture is part of the downstream readout and has the same
    25-state interface across XLS-R-300M and WavLM-Large.  Absolute or XOR
    encoded ``frontend.model`` tensors are never transferred across frontends.
    """
    state = dict(blob["model"])
    if reader_only:
        state = {
            key: value for key, value in state.items()
            if not key.startswith("frontend.model.")
        }
        if not state:
            raise ValueError("--init-reader-only found no downstream tensors")
    elif blob.get("encoder_encoding") is not None or blob.get("encoder_xor") is not None:
        if blob.get("encoder_encoding") != "fp16_xor_pretrained_v1":
            raise ValueError(
                f"unsupported initialization encoding: {blob.get('encoder_encoding')!r}"
            )
        encoded = blob.get("encoder_xor")
        declared = tuple(blob.get("trained_encoder_keys", ()))
        if not isinstance(encoded, dict) or set(encoded) != set(declared):
            raise ValueError("encoded initialization tensors disagree with metadata")
        if model is None:
            raise ValueError("encoded initialization requires the target model")
        base = model.state_dict()
        for name, xor_bits in encoded.items():
            if name not in base:
                raise ValueError(f"encoded initialization tensor is absent: {name}")
            base_half = base[name].detach().cpu().contiguous().half()
            bits = xor_bits.detach().cpu().contiguous()
            if bits.dtype != torch.int16 or bits.shape != base_half.shape:
                raise ValueError(f"malformed encoded initialization tensor: {name}")
            state[name] = torch.bitwise_xor(
                base_half.view(torch.int16), bits
            ).view(torch.float16)
    return {
        key: value.float() if value.is_floating_point() else value
        for key, value in state.items()
    }


def optimizer_parameter_groups(
    model: nn.Module,
    lr_ssl: float,
    lr_head: float,
    lr_layer_mix: float = 0.0,
) -> list[dict]:
    """Build disjoint optimizer groups, optionally accelerating layer logits."""
    groups = [dict(group) for group in model.param_groups(lr_ssl, lr_head)]
    if lr_layer_mix > 0:
        layer_weights = getattr(getattr(model, "frontend", None), "layer_weights", None)
        if not isinstance(layer_weights, nn.Parameter) or not layer_weights.requires_grad:
            raise ValueError(
                "--lr-layer-mix requires a trainable learned layer mixture"
            )
        found = 0
        filtered: list[dict] = []
        for group in groups:
            parameters = list(group["params"])
            kept = []
            for parameter in parameters:
                if parameter is layer_weights:
                    found += 1
                else:
                    kept.append(parameter)
            if kept:
                updated = dict(group)
                updated["params"] = kept
                filtered.append(updated)
        if found != 1:
            raise RuntimeError(
                f"layer mixture appeared in {found} base optimizer groups, expected one"
            )
        groups = filtered + [{
            "params": [layer_weights],
            "lr": lr_layer_mix,
            "name": "layer-mixture",
        }]

    grouped = [parameter for group in groups for parameter in group["params"]]
    grouped_ids = [id(parameter) for parameter in grouped]
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != expected_ids:
        raise RuntimeError("optimizer groups do not partition trainable parameters exactly")
    return groups


def layer_mixture_stats(model: nn.Module) -> dict[str, float | int] | None:
    """Return interpretable concentration diagnostics for an all-state mixture."""
    logits = getattr(getattr(model, "frontend", None), "layer_weights", None)
    if not isinstance(logits, torch.Tensor):
        return None
    values = logits.detach().float().cpu()
    probability = torch.softmax(values, dim=0)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return {
        "logit_spread": float(values.max() - values.min()),
        "effective_layers": float(torch.exp(entropy)),
        "max_probability": float(probability.max()),
        "argmax_layer": int(probability.argmax()),
    }


def select_anchor_second_pass_parameters(model: nn.Module) -> tuple[str, ...]:
    """Freeze everything except the inherited anchor-conditioned refinement pass.

    This deliberately operates on parameter names rather than module types: the
    exact treatment is the three restored checkpoint namespaces, and a missing
    namespace must fail loudly instead of silently changing experimental scope.
    """
    selected: list[str] = []
    seen_prefixes: set[str] = set()
    for name, parameter in model.named_parameters():
        prefix = next((p for p in ANCHOR_SECOND_PASS_PREFIXES if name.startswith(p)), None)
        parameter.requires_grad_(prefix is not None)
        if prefix is not None:
            selected.append(name)
            seen_prefixes.add(prefix)
    missing = sorted(set(ANCHOR_SECOND_PASS_PREFIXES) - seen_prefixes)
    if missing:
        raise ValueError(f"anchor-second-pass scope missing parameter namespaces: {missing}")
    return tuple(selected)


def select_anchor_prepass_parameters(model: nn.Module) -> tuple[str, ...]:
    """Freeze everything except the inherited path that constructs pass-one states.

    Together with :func:`select_anchor_second_pass_parameters`, this forms an
    exact partition of the restored 4.61M-parameter reader.  The learned layer
    mixture, encoder, anchor-conditioned refinement, and dead position head are
    deliberately outside this scope.
    """
    selected: list[str] = []
    seen_prefixes: set[str] = set()
    seen_exact: set[str] = set()
    for name, parameter in model.named_parameters():
        prefix = next((p for p in ANCHOR_PREPASS_PREFIXES if name.startswith(p)), None)
        exact = name in ANCHOR_PREPASS_EXACT_NAMES
        parameter.requires_grad_(prefix is not None or exact)
        if prefix is not None:
            selected.append(name)
            seen_prefixes.add(prefix)
        elif exact:
            selected.append(name)
            seen_exact.add(name)
    missing = sorted(
        (set(ANCHOR_PREPASS_PREFIXES) - seen_prefixes)
        | (set(ANCHOR_PREPASS_EXACT_NAMES) - seen_exact)
    )
    if missing:
        raise ValueError(f"anchor-prepass scope missing parameter namespaces: {missing}")
    return tuple(selected)


def select_encoder_only_parameters(model: nn.Module) -> tuple[str, ...]:
    """Freeze the inherited reader and retain exactly the declared SSL suffix."""
    frontend = getattr(model, "frontend", None)
    encoder = getattr(getattr(frontend, "model", None), "encoder", None)
    layers = getattr(encoder, "layers", None)
    unfreeze_top = int(getattr(frontend, "unfreeze_top", 0))
    if layers is None or unfreeze_top <= 0:
        raise ValueError("encoder-only scope requires a positive selective encoder suffix")
    expected_blocks = set(range(len(layers) - unfreeze_top, len(layers)))
    selected: list[str] = []
    actual_blocks: set[int] = set()
    prefix = "frontend.model.encoder.layers."
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefix) and parameter.requires_grad
        parameter.requires_grad_(trainable)
        if not trainable:
            continue
        block_text = name[len(prefix):].split(".", 1)[0]
        if not block_text.isdigit():
            raise ValueError(f"cannot resolve encoder block from trainable tensor {name}")
        actual_blocks.add(int(block_text))
        selected.append(name)
    if actual_blocks != expected_blocks:
        raise ValueError(
            "encoder-only block mismatch: "
            f"expected={sorted(expected_blocks)}, actual={sorted(actual_blocks)}"
        )
    if not selected:
        raise ValueError("encoder-only scope selected no trainable tensors")
    return tuple(selected)


def replace_fine_classifiers_for_native_only(model: nn.Module) -> tuple[str, ...]:
    """Freeze exactly the final classifiers replaced by the native coarse head."""
    replaced: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("out2.") or name.startswith("position_out."):
            parameter.requires_grad_(False)
            replaced.append(name)
    if set(replaced) != set(NATIVE_REPLACED_CLASSIFIER_KEYS):
        raise RuntimeError(f"native-only replaced-classifier scope changed: {replaced}")
    return tuple(replaced)


def validate_selected_gradients(
    model: nn.Module, selected: tuple[str, ...], scope_name: str = "scoped"
) -> None:
    """Reject dead or non-finite tensors before the first scoped optimizer step."""
    named = dict(model.named_parameters())
    failures = []
    for name in selected:
        grad = named[name].grad
        if grad is None:
            failures.append(f"{name}:missing")
        elif not torch.isfinite(grad).all():
            failures.append(f"{name}:nonfinite")
        elif grad.detach().float().norm().item() == 0.0:
            failures.append(f"{name}:zero")
    if failures:
        raise RuntimeError(f"invalid {scope_name} gradients: {failures[:5]}")


def scheduled_position_weight(
    weight: float, stop_fraction: float, batch_index: int, total_batches: int
) -> float:
    """Apply position supervision through an inclusive fraction of training."""
    if not 0.0 <= stop_fraction <= 1.0:
        raise ValueError("position stop fraction must be in [0, 1]")
    if not 1 <= batch_index <= total_batches:
        raise ValueError("batch index must be within the training run")
    return weight if batch_index / total_batches <= stop_fraction else 0.0


def segment_loss(out: torch.Tensor, lab: torch.Tensor, args, criterion) -> torch.Tensor:
    """Objective on the final (B, S, 2) logits. See --loss."""
    logits = out.float().reshape(-1, 2)
    y = lab.reshape(-1)
    valid = y != -100
    if args.loss == "ce":
        return criterion(logits, y)
    if args.loss == "rank":
        d = (logits[:, 1] - logits[:, 0])[valid]
        yv = y[valid]
        pos, neg = d[yv == 1], d[yv == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return criterion(logits, y)
        if pos.numel() > args.rank_pairs:
            pos = pos[torch.randperm(pos.numel(), device=pos.device)[: args.rank_pairs]]
        if neg.numel() > args.rank_pairs:
            neg = neg[torch.randperm(neg.numel(), device=neg.device)[: args.rank_pairs]]
        # pairwise logistic: every (bonafide, spoof) pair pooled over the batch, as the metric pools
        rank = F.softplus(-(pos[:, None] - neg[None, :])).mean()
        return rank + args.rank_ce_weight * criterion(logits, y)
    if args.loss == "boundary":
        y2 = lab.clamp(min=0)
        v2 = lab != -100
        change = torch.zeros_like(v2)
        change[:, 1:] = (y2[:, 1:] != y2[:, :-1]) & v2[:, 1:] & v2[:, :-1]
        k = args.boundary_k
        near = F.max_pool1d(change.float()[:, None], kernel_size=2 * k + 1, stride=1, padding=k)[:, 0] > 0
        if getattr(args, "boundary_soft", False):
            dist = torch.full(change.shape, float(10 * k), device=change.device)
            for d in range(k, -1, -1):          # shrinking windows: the last hit sets the true distance
                hit = F.max_pool1d(change.float()[:, None], kernel_size=2 * d + 1, stride=1, padding=d)[:, 0] > 0
                dist = torch.where(hit, torch.full_like(dist, float(d)), dist)
            near_w = torch.exp(-0.5 * (dist / max(1.0, k / 2.0)) ** 2)
            w = (1.0 + args.boundary_beta * near_w).reshape(-1)
        else:
            w = (1.0 + args.boundary_beta * near.float()).reshape(-1)
        per = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
        return (per * w)[valid].sum() / w[valid].sum()
    if args.loss == "focal":
        per = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
        pt = torch.exp(-per)
        return (((1 - pt) ** args.focal_gamma) * per)[valid].mean()
    if args.loss == "lovasz":
        # Lovasz hinge on the SPOOF class (a convex surrogate of 1 - IoU of the spoof spans), pooled over
        # the batch's valid segments, + 0.1 CE. Credits coherent spans and penalises ragged ones directly.
        d = (logits[:, 0] - logits[:, 1])[valid]          # spoof-positive margin
        yv = (y[valid] == 0).float()
        signs = 2.0 * yv - 1.0
        errors = (1.0 - d * signs)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = yv[perm]
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jacc = 1.0 - inter / union.clamp(min=1.0)
        if jacc.numel() > 1:
            jacc[1:] = jacc[1:] - jacc[:-1]
        lov = torch.dot(F.relu(errors_sorted), jacc)
        return lov + 0.1 * criterion(logits, y)
    if args.loss == "ocsoftmax":
        # OC-Softmax per frame: s = cos(y2, w) recovered from the logit difference d = alpha * s
        d = (logits[:, 1] - logits[:, 0])[valid]
        s_cos = d / args.oc_alpha
        yv = y[valid]
        margin = torch.where(yv == 1, args.oc_m0 - s_cos, s_cos - args.oc_m1)
        return F.softplus(args.oc_alpha * margin).mean()
    if args.loss == "rank+boundary":
        a = args.loss; args.loss = "rank"; l1 = segment_loss(out, lab, args, criterion)
        args.loss = "boundary"; l2 = segment_loss(out, lab, args, criterion); args.loss = a
        return l1 + 0.5 * l2
    raise ValueError(args.loss)


def angular_one_class_loss(
    states: torch.Tensor,
    classifier_weight: torch.Tensor,
    lab: torch.Tensor,
    alpha: float = 20.0,
    bona_margin: float = 0.9,
    spoof_margin: float = 0.2,
) -> torch.Tensor:
    """Faithful one-class angular loss on valid segment states.

    ``classifier_weight[1] - classifier_weight[0]`` is the inherited
    bona-fide score direction because this repository emits
    ``logit[1] - logit[0]``.  Both it and every state are L2-normalized before
    applying the asymmetric OC-Softmax margins.  This is intentionally
    separate from the historical ``--loss ocsoftmax`` logit-margin
    placeholder, which has no normalized embedding geometry.
    """
    if states.ndim != 3 or lab.shape != states.shape[:2]:
        raise ValueError(
            f"expected states (B,S,D) and matching labels, got "
            f"{tuple(states.shape)} and {tuple(lab.shape)}"
        )
    if classifier_weight.shape != (2, states.shape[-1]):
        raise ValueError(
            f"expected classifier weight (2,{states.shape[-1]}), got "
            f"{tuple(classifier_weight.shape)}"
        )
    if alpha <= 0:
        raise ValueError("angular OC alpha must be positive")
    if not -1.0 <= spoof_margin < bona_margin <= 1.0:
        raise ValueError("angular OC margins must satisfy -1 <= spoof < bona <= 1")

    valid = lab != -100
    if not valid.any():
        # Preserve a differentiable zero for defensive all-padding batches.
        return states.float().sum() * 0.0 + classifier_weight.float().sum() * 0.0
    labels = lab[valid]
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("angular OC labels must be 0=spoof or 1=bonafide")

    embeddings = F.normalize(states.float()[valid], dim=-1)
    bona_direction = F.normalize(
        (classifier_weight[1] - classifier_weight[0]).float(), dim=0
    )
    cosine = (embeddings * bona_direction).sum(dim=-1).clamp(-1.0, 1.0)
    margin_violation = torch.where(
        labels == 1,
        float(bona_margin) - cosine,
        cosine - float(spoof_margin),
    )
    return F.softplus(float(alpha) * margin_violation).mean()


def multires_loss(out: torch.Tensor, lab: torch.Tensor, tau: float) -> torch.Tensor:
    """Auxiliary coarse-grid loss. d = logit(bona) - logit(spoof) per 20 ms segment;
    a block's score is a soft-min of d over its segments (a single spoof segment makes
    the block spoof, exactly the rule PS uses to derive its 40..640 ms labels)."""
    d = (out[..., 1] - out[..., 0]).float()                       # (B, S)
    valid = lab != -100
    y = (lab == 1).float()
    total = d.new_zeros(())
    for r in MULTIRES_BLOCKS:
        S = d.shape[1]; pad = (-S) % r
        dp = F.pad(d, (0, pad), value=1e4).view(d.shape[0], -1, r)          # pad = "bonafide, ignore"
        vp = F.pad(valid, (0, pad), value=False).view(d.shape[0], -1, r)
        yp = F.pad(y, (0, pad), value=1.0).view(d.shape[0], -1, r)
        block_valid = vp.any(-1)
        if not block_valid.any():
            continue
        # soft-min over the block; padded / invalid segments are pushed to +inf so they never win
        dm = dp.masked_fill(~vp, 1e4)
        block_score = -tau * torch.logsumexp(-dm / tau, dim=-1)
        block_y = (yp.masked_fill(~vp, 1.0).min(-1).values)
        total = total + F.binary_cross_entropy_with_logits(block_score[block_valid], block_y[block_valid])
    return total / len(MULTIRES_BLOCKS)


def native_block_targets(
    lab: torch.Tensor, block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bonafide-positive any-spoof labels on aligned coarse blocks."""
    if lab.ndim != 2 or block_size < 1:
        raise ValueError("native labels must be rank two and block size positive")
    valid = lab != -100
    positions = torch.arange(lab.shape[1], device=lab.device)[None, :]
    lengths = valid.sum(dim=1)
    if not torch.equal(valid, positions < lengths[:, None]):
        raise ValueError("valid native labels must form a contiguous prefix")
    pad = (-lab.shape[1]) % block_size
    bona = (lab == 1).masked_fill(~valid, True)
    if pad:
        bona = F.pad(bona, (0, pad), value=True)
        valid = F.pad(valid, (0, pad), value=False)
    bona = bona.view(lab.shape[0], -1, block_size)
    valid = valid.view(lab.shape[0], -1, block_size)
    return bona.all(dim=-1).float(), valid.any(dim=-1)


def native_bce_loss(
    model: nn.Module, lab: torch.Tensor, resolution_ms: int = 160,
) -> tuple[torch.Tensor, int, int]:
    """BCE on a model-emitted native grid, with fail-closed class authority."""
    native_scores = getattr(model, "_aux", {}).get("native_scores")
    native_valid = getattr(model, "_aux", {}).get("native_valid")
    if not isinstance(native_scores, dict) or not isinstance(native_valid, dict):
        raise RuntimeError("native objective requires model-emitted native scores and masks")
    if resolution_ms not in native_scores or resolution_ms not in native_valid:
        raise RuntimeError(f"model did not emit native {resolution_ms} ms scores")
    block_size = resolution_ms // 20
    target, target_valid = native_block_targets(lab, block_size)
    prediction = native_scores[resolution_ms]
    usable = native_valid[resolution_ms] & target_valid
    if prediction.shape != target.shape or usable.shape != target.shape:
        raise RuntimeError(
            "native score/target mismatch: "
            f"scores={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    truth = target[usable]
    n_bonafide = int((truth == 1).sum())
    n_spoof = int((truth == 0).sum())
    if not n_bonafide or not n_spoof:
        raise RuntimeError(
            "native training batch lacks one block class: "
            f"bonafide={n_bonafide}, spoof={n_spoof}"
        )
    return (
        F.binary_cross_entropy_with_logits(prediction[usable], truth),
        n_bonafide,
        n_spoof,
    )


def enter_train_mode(
    model: nn.Module,
    adapter_only: bool = False,
    encoder_only: bool = False,
) -> None:
    """Enable cuDNN RNN backward while keeping a frozen parent deterministic."""
    if encoder_only:
        # Gradients must traverse the fixed recurrent reader into XLS-R. cuDNN
        # RNN backward requires training=True, so enable only recurrent modules;
        # their internal dropout is explicitly zeroed during scope setup.
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.RNNBase):
                module.train()
        return
    model.train()
    if adapter_only:
        # cuDNN requires recurrent modules to have training=True for backward.
        # Ordinary Dropout modules can still be disabled independently; the
        # two-layer pass-1 LSTM has its internal probability set to zero once
        # during adapter-only setup below.
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=str(Path(__file__).parent / "manifests"))
    ap.add_argument("--taskdata", default=TASKDATA)
    ap.add_argument("--name", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument(
        "--frontend-revision", default=None,
        help="optional immutable Hugging Face revision for frontend reconstruction",
    )
    ap.add_argument("--layer", type=int, default=-1, help="-1 = learned weighted sum of all layers")
    ap.add_argument("--n-layers-keep", type=int, default=0, help="0 = all 24 transformer layers")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument(
        "--unfreeze-top", type=int, default=0,
        help="with --freeze, re-enable the top N XLS-R transformer blocks",
    )
    ap.add_argument("--head", default="bilstm", choices=("bilstm", "conv"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--layerdrop", type=float, default=0.05)
    ap.add_argument("--feat-norm", default="none", choices=("none", "mean", "meanstd", "concat"))
    ap.add_argument("--layer-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict the learned layer mixture to hidden states LO..HI (0 = CNN output, 24 = last)")
    ap.add_argument("--no-wav-norm", action="store_true", help="disable per-utterance waveform normalisation (XLS-R expects it on)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--schedule-epochs", type=int, default=0,
                    help="cosine horizon in epochs (0 = --epochs); permits a long-horizon "
                         "schedule to be observed for fewer training epochs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop", type=int, default=200, help="training crop in 20 ms segments")
    ap.add_argument(
        "--crop-alignment", type=int, default=1,
        help="restrict random crop starts to this utterance-origin frame phase; "
             "must divide --crop",
    )
    ap.add_argument("--lr-ssl", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument(
        "--lr-layer-mix", type=float, default=0.0,
        help="optional dedicated LR for learned all-state mixture logits; 0 keeps "
             "them in the ordinary head group",
    )
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--grad-scale-init", type=float, default=65536.0,
        help="initial CUDA GradScaler value; lower it when the audited first backward "
             "shows finite losses but fp16 gradient overflow",
    )
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=3000)
    ap.add_argument("--eval-every", type=float, default=1.0, help="epochs between dev evals")
    ap.add_argument("--augment", default="none", help="none | name understood by pslps.augment.build")
    ap.add_argument("--aug-args", default="{}", help="JSON kwargs for the augmenter")
    ap.add_argument("--arch", default="anchor",
                    choices=("frame", "anchor", "boundary_unet", "position_anchor", "native_pooled"),
                    help="frame = independent per-segment head (XLSRLocalizer); "
                         "anchor = anchor-relative read-out (AnchorLocalizer); "
                         "boundary_unet = sparse-layer boundary-first reader; "
                         "position_anchor = anchor reader with SAL-style position auxiliary loss; "
                         "native_pooled = position-anchor plus a direct attentive 160 ms branch")
    ap.add_argument("--native-weight", type=float, default=0.0,
                    help="weight of direct 160 ms BCE for --arch native_pooled")
    ap.add_argument(
        "--native-authority-min", type=float, default=0.0,
        help="fail before the first optimizer step unless native BCE contributes at least "
             "this fraction of the measured pre-update total loss",
    )
    ap.add_argument(
        "--native-authority-max", type=float, default=1.0,
        help="fail before the first optimizer step unless native BCE contributes at most "
             "this fraction of the measured pre-update total loss",
    )
    ap.add_argument("--native-attention-hidden", type=int, default=64)
    ap.add_argument("--native-classifier-hidden", type=int, default=128)
    ap.add_argument("--native-dropout", type=float, default=0.1)
    ap.add_argument("--native-trunk-layers", type=int, default=0)
    ap.add_argument("--native-trunk-bottleneck", type=int, default=64)
    ap.add_argument(
        "--native-random-output-init", action="store_true",
        help="retain the fresh native classifier's random final layer instead of "
             "zero-initializing it; required for first-step end-to-end authority",
    )
    ap.add_argument(
        "--native-only", action="store_true",
        help="replace the final fine classifier and optimize only native 160 ms BCE; "
             "the resulting checkpoint must be deployed with a separate fine model",
    )
    ap.add_argument(
        "--fine-deployment-ckpt", default="",
        help="committed immutable 20 ms checkpoint used alongside a --native-only model",
    )
    ap.add_argument("--unet-boundary-weight", type=float, default=0.5,
                    help="weight of exact-transition pyramid BCE for boundary_unet")
    ap.add_argument("--unet-probability-gate", action="store_true",
                    help="gate every boundary-to-class route by predicted seam probability")
    ap.add_argument("--unet-symmetric-boundary", action="store_true",
                    help="supervise both cells adjacent to a label transition")
    ap.add_argument("--position-weight", type=float, default=0.1,
                    help="weight of the eight-way class-by-position auxiliary CE")
    ap.add_argument("--position-stop-frac", type=float, default=1.0,
                    help="inclusive training fraction through which position loss is active")
    ap.add_argument("--init-ckpt", default="",
                    help="initialize trainable weights from a compatible partial/full checkpoint")
    ap.add_argument(
        "--init-reader-only", action="store_true",
        help="with --init-ckpt, load every downstream/readout tensor including the "
             "layer mixture but never transfer frontend.model tensors (for SSL-family pivots)",
    )
    ap.add_argument("--freeze-layer-mixture", action="store_true",
                    help="retain checkpoint layer-mixture coefficients without updating them")
    ap.add_argument("--adapter-dim", type=int, default=0,
                    help="enable zero-output multi-scale adapters on returned XLS-R states")
    ap.add_argument("--adapter-kernels", type=int, nargs="+", default=(3, 7, 15, 23))
    ap.add_argument("--adapter-layers", type=int, nargs="+", default=None,
                    help="returned hidden-state indices to adapt (default: all 25)")
    ap.add_argument("--adapter-only", action="store_true",
                    help="freeze the inherited mixture/read-out and train only temporal adapters")
    ap.add_argument("--anchor-second-pass-only", action="store_true",
                    help="freeze every restored tensor except anchor_in/rnn2/out2")
    ap.add_argument("--anchor-prepass-only", action="store_true",
                    help="freeze every restored tensor except norm/norm2/proj/rnn/out/log_tau")
    ap.add_argument(
        "--encoder-only", action="store_true",
        help="freeze the inherited reader and optimize only blocks selected by --unfreeze-top",
    )
    ap.add_argument(
        "--audit-trainable-gradients", action="store_true",
        help="before the first optimizer step, require every loss-reachable trainable tensor "
             "to have a finite nonzero gradient",
    )
    ap.add_argument(
        "--probe-only", action="store_true",
        help="run one audited forward/backward, report loss authority, and exit before "
             "any optimizer step or checkpoint write",
    )
    ap.add_argument("--span-frame-weight", type=float, default=0.5, help="aux CE on raw frame emissions")
    ap.add_argument("--span-boundary-weight", type=float, default=0.5, help="aux BCE on change points")
    ap.add_argument("--span-dilate", type=int, default=1)
    ap.add_argument("--span-feat-diff", action="store_true", help="generic frozen-feature discontinuity cue into the change-point head")
    ap.add_argument("--anchor-mode", default="soft", choices=("soft", "topk"))
    ap.add_argument("--anchor-topk", type=float, default=0.25)
    ap.add_argument("--anchor-source", default="state", choices=("state", "feats"))
    ap.add_argument("--loss", default="ce", choices=("ce", "rank", "boundary", "focal", "lovasz", "rank+boundary", "ocsoftmax"),
                    help="per-segment objective on the FINAL logits: ce = cross-entropy (baseline); "
                         "rank = batch-pooled pairwise logistic ranking loss on d = l1 - l0 (an AUC surrogate, "
                         "i.e. what segment EER measures) + rank_ce_weight * CE; boundary = CE with frames within "
                         "boundary_k of a label transition up-weighted by (1 + boundary_beta); focal = focal CE")
    ap.add_argument("--rank-ce-weight", type=float, default=0.1)
    ap.add_argument("--rank-pairs", type=int, default=4096, help="max bonafide / spoof frames sampled per batch for the pairwise loss")
    ap.add_argument("--boundary-k", type=int, default=5)
    ap.add_argument("--boundary-beta", type=float, default=2.0)
    ap.add_argument("--boundary-soft", action="store_true",
                    help="Gaussian falloff of the extra weight with distance to the nearest transition (sigma = k/2) instead of a hard window")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--oc-alpha", type=float, default=20.0)
    ap.add_argument("--oc-m0", type=float, default=0.9, help="bonafide margin (cos >= m0)")
    ap.add_argument("--oc-m1", type=float, default=0.2, help="spoof margin (cos <= m1)")
    ap.add_argument(
        "--oc-aux-weight", type=float, default=0.0,
        help="weight of faithful normalized one-class angular loss on anchor pass-two states; "
             "keeps the inherited focal logits as the inference score",
    )
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay over this run's own trainable weights (0 = off); "
                         "the EMA copy is evaluated and saved alongside (<run>.ep<k>.ema.pt)")
    ap.add_argument("--multires-weight", type=float, default=0.0,
                    help="weight of the auxiliary multi-resolution loss (blocks of 2,4,8,16,32 segments = 40..640 ms); "
                         "block label = bonafide iff all its 20 ms labels are bonafide (matches PS coarse labels)")
    ap.add_argument("--multires-tau", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    schedule_epochs = resolve_schedule_epochs(args.epochs, args.schedule_epochs)
    if args.unfreeze_top < 0:
        raise ValueError("--unfreeze-top must be nonnegative")
    if args.unfreeze_top and not args.freeze:
        raise ValueError("--unfreeze-top requires --freeze")
    if args.crop_alignment < 1 or args.crop % args.crop_alignment:
        raise ValueError("--crop-alignment must be positive and divide --crop")
    if args.native_weight < 0:
        raise ValueError("--native-weight must be nonnegative")
    if not math.isfinite(args.grad_scale_init) or args.grad_scale_init <= 0:
        raise ValueError("--grad-scale-init must be finite and positive")
    if not math.isfinite(args.lr_layer_mix) or args.lr_layer_mix < 0:
        raise ValueError("--lr-layer-mix must be finite and nonnegative")
    if args.freeze_layer_mixture and args.lr_layer_mix:
        raise ValueError("--lr-layer-mix conflicts with --freeze-layer-mixture")
    if not 0.0 <= args.native_authority_min <= args.native_authority_max <= 1.0:
        raise ValueError("native authority bounds must satisfy 0 <= min <= max <= 1")
    if args.arch == "native_pooled":
        if args.native_weight <= 0:
            raise ValueError("--arch native_pooled requires --native-weight > 0")
        if args.crop_alignment % 8:
            raise ValueError("native 160 ms training requires --crop-alignment divisible by 8")
    elif (args.native_weight or args.native_authority_min != 0.0
          or args.native_authority_max != 1.0):
        raise ValueError("native loss and authority options require --arch native_pooled")
    if args.init_reader_only and not args.init_ckpt:
        raise ValueError("--init-reader-only requires --init-ckpt")
    if args.native_only:
        if args.arch != "native_pooled" or not args.init_ckpt:
            raise ValueError("--native-only requires --arch native_pooled and --init-ckpt")
        if not args.native_random_output_init:
            raise ValueError(
                "--native-only requires --native-random-output-init so update one reaches the base"
            )
        if (args.native_weight != 1.0 or args.native_authority_min != 1.0
                or args.native_authority_max != 1.0):
            raise ValueError(
                "--native-only requires native weight/min/max = 1/1/1; it is the objective, not an auxiliary"
            )
        if args.position_weight or args.oc_aux_weight or args.multires_weight:
            raise ValueError(
                "--native-only replaces fine supervision and requires all fine/coarse auxiliaries off"
            )
        fine_checkpoint = Path(__file__).parent / "ckpt" / args.fine_deployment_ckpt
        if (Path(args.fine_deployment_ckpt).name != args.fine_deployment_ckpt
                or not fine_checkpoint.is_file()):
            raise ValueError(
                "--native-only requires --fine-deployment-ckpt as a local checkpoint basename"
            )
    elif args.fine_deployment_ckpt:
        raise ValueError("--fine-deployment-ckpt is only valid with --native-only")
    if args.probe_only and (not args.audit_trainable_gradients or args.accum != 1):
        raise ValueError("--probe-only requires --audit-trainable-gradients and --accum 1")
    if args.encoder_only and not args.unfreeze_top:
        raise ValueError("--encoder-only requires --unfreeze-top > 0")
    if args.oc_aux_weight < 0:
        raise ValueError("--oc-aux-weight must be nonnegative")
    if args.oc_aux_weight > 0:
        if args.arch not in ("anchor", "position_anchor", "native_pooled"):
            raise ValueError("--oc-aux-weight requires an anchor architecture")
        if args.loss != "focal":
            raise ValueError("--oc-aux-weight requires --loss focal for the registered treatment")
        if args.oc_alpha <= 0 or not -1.0 <= args.oc_m1 < args.oc_m0 <= 1.0:
            raise ValueError(
                "angular OC requires alpha > 0 and -1 <= m1 < m0 <= 1"
            )

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda"
    man, td = Path(args.manifests), Path(args.taskdata)
    tr_items = remap(read_manifest(man / "ps_train.tsv"))
    va_items = remap(read_manifest(man / "ps_dev.tsv"))
    rng = random.Random(0)
    rng.shuffle(va_items)                     # fixed random dev slice, not the first N ids
    if args.limit_train:
        tr_items = tr_items[: args.limit_train]
    if args.limit_val:
        va_items = va_items[: args.limit_val]
    va_items.sort(key=lambda u: u.n_segments)  # length-bucketed batches for eval
    tr_lab = load_label_npz(td / "train_labels" / "ps_train.npz")
    va_lab = load_label_npz(td / "train_labels" / "ps_dev.npz")

    augment = None
    if args.augment != "none":
        from spanmark.audio.augment import build
        augment = build(args.augment, tr_items, tr_lab, **json.loads(args.aug_args))

    # TRAINING DATA RULE (task.yaml): PartialSpoof train only. Everything else is derived from it.
    tl = DataLoader(CropDataset(
                        tr_items, tr_lab, args.crop, augment,
                        crop_alignment=args.crop_alignment,
                    ), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0)
    hl = None
    vl = DataLoader(SegmentDataset(va_items, "", va_lab), batch_size=8, shuffle=False,
                    num_workers=args.workers, collate_fn=collate, pin_memory=True)

    selected_layer = None if args.layer < 0 else args.layer
    if args.arch == "boundary_unet":
        if selected_layer is None:
            raise ValueError("boundary_unet requires an explicit --layer")
        if not args.freeze:
            raise ValueError("boundary_unet requires --freeze")
        model_args = dict(name=args.name, revision=args.frontend_revision,
                          layer=selected_layer, freeze=True,
                          n_layers_keep=args.n_layers_keep or selected_layer,
                          hidden=args.hidden, dropout=args.dropout,
                          layerdrop=0.0, wav_norm=not args.no_wav_norm,
                          probability_gate=args.unet_probability_gate,
                          symmetric_boundary=args.unet_symmetric_boundary)
        model = BoundaryUNetLocalizer(**model_args).to(device)
    else:
        model_args = dict(name=args.name, revision=args.frontend_revision,
                          layer=selected_layer,
                          freeze=args.freeze, n_layers_keep=args.n_layers_keep or None,
                          hidden=args.hidden, dropout=args.dropout, head=args.head,
                          layerdrop=args.layerdrop, feat_norm=args.feat_norm,
                          wav_norm=not args.no_wav_norm,
                          layer_range=tuple(args.layer_range) if args.layer_range else None,
                          adapter_dim=args.adapter_dim,
                          adapter_kernels=tuple(args.adapter_kernels),
                          adapter_layers=(tuple(args.adapter_layers)
                                          if args.adapter_layers is not None else None),
                          unfreeze_top=args.unfreeze_top)
        if args.arch in ("anchor", "position_anchor"):
            anchor_cls = PositionAnchorLocalizer if args.arch == "position_anchor" else AnchorLocalizer
            model = anchor_cls(**model_args, anchor_mode=args.anchor_mode,
                               anchor_topk=args.anchor_topk,
                               anchor_source=args.anchor_source).to(device)
        elif args.arch == "native_pooled":
            model = NativePooledLocalizer(
                **model_args,
                anchor_mode=args.anchor_mode,
                anchor_topk=args.anchor_topk,
                anchor_source=args.anchor_source,
                native_resolutions_ms=(160,),
                native_attention_hidden=args.native_attention_hidden,
                native_classifier_hidden=args.native_classifier_hidden,
                native_dropout=args.native_dropout,
                native_trunk_layers=args.native_trunk_layers,
                native_trunk_bottleneck=args.native_trunk_bottleneck,
                native_zero_output_init=not args.native_random_output_init,
            ).to(device)
        else:
            model = XLSRLocalizer(**model_args).to(device)
    if args.arch == "boundary_unet":
        saved_args = dict(model_args)
    elif args.arch in ("span", "refinespan"):
        saved_args = dict(model_args, boundary_dilate=args.span_dilate, boundary_feat_diff=args.span_feat_diff)
    elif args.arch in ("anchorspan", "multibranch"):
        saved_args = dict(model_args, boundary_dilate=args.span_dilate, anchor_mode=args.anchor_mode,
                          anchor_topk=args.anchor_topk, anchor_source=args.anchor_source)
    elif args.arch in ("anchor", "position_anchor", "dynanchor"):
        saved_args = dict(model_args, anchor_mode=args.anchor_mode, anchor_topk=args.anchor_topk,
                          anchor_source=args.anchor_source)
    elif args.arch == "native_pooled":
        saved_args = dict(
            model_args,
            anchor_mode=args.anchor_mode,
            anchor_topk=args.anchor_topk,
            anchor_source=args.anchor_source,
            native_resolutions_ms=(160,),
            native_attention_hidden=args.native_attention_hidden,
            native_classifier_hidden=args.native_classifier_hidden,
            native_dropout=args.native_dropout,
            native_trunk_layers=args.native_trunk_layers,
            native_trunk_bottleneck=args.native_trunk_bottleneck,
            native_zero_output_init=not args.native_random_output_init,
        )
    elif args.arch == "ocanchor":
        saved_args = dict(model_args, anchor_mode=args.anchor_mode, anchor_topk=args.anchor_topk,
                          anchor_source=args.anchor_source, oc_alpha=args.oc_alpha)
    else:
        saved_args = dict(model_args)
    arch_name = {"boundary_unet": "boundary_unet_localizer",
                 "position_anchor": "position_anchor_localizer",
                 "span": "span_localizer", "crf": "crf_localizer", "refine": "refine_localizer",
                 "refinespan": "refinespan_localizer", "anchor": "anchor_localizer",
                 "anchorspan": "anchorspan_localizer", "dynanchor": "dynanchor_localizer",
                 "dualanchor": "dualanchor_localizer", "multibranch": "multibranch_localizer", "ocanchor": "ocanchor_localizer"}.get(args.arch, "xlsr_localizer")
    if args.arch == "native_pooled":
        arch_name = "native_pooled_localizer"

    if args.init_ckpt:
        init_blob = load_checkpoint_blob(args.init_ckpt)
        init_state = initialization_state(
            init_blob, args.init_reader_only, model=model
        )
        if args.init_reader_only and any(
            key.startswith("frontend.model.") for key in init_state
        ):
            raise RuntimeError("reader-only initialization retained an SSL tensor")
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        allowed_missing = [
            key for key in missing
            if key.startswith("frontend.model.")
            or key.startswith("frontend.adapters.")
            or (args.arch in ("position_anchor", "native_pooled")
                and key.startswith("position_out."))
            or (args.native_only and key.startswith("native_pool."))
        ]
        allowed_unexpected = [
            key for key in unexpected
            if args.arch == "anchor" and key.startswith("position_out.")
        ]
        bad_missing = sorted(set(missing) - set(allowed_missing))
        bad_unexpected = sorted(set(unexpected) - set(allowed_unexpected))
        if bad_unexpected or bad_missing:
            raise ValueError(
                f"incompatible --init-ckpt: unexpected={bad_unexpected[:5]} "
                f"missing={bad_missing[:5]}"
            )
        new_trainable = [key for key in allowed_missing if not key.startswith("frontend.model.")]
        print(f"initialized from {args.init_ckpt}; "
              f"restored {len(init_state)} tensors"
              f"{' (reader only)' if args.init_reader_only else ''}; "
              f"new trainable keys: {len(new_trainable)} ({new_trainable[:4]}...)",
              flush=True)
        if allowed_unexpected:
            print(f"dropped inference-inert source keys: {allowed_unexpected}", flush=True)
    if args.native_only:
        # The published resolution-specific recipe replaces the base model's
        # final classifier.  Keep the old fine and position classifiers only so
        # the parent state remains auditable; neither participates in this
        # dedicated model's loss or deployment output.
        replaced = replace_fine_classifiers_for_native_only(model)
        print(
            "native-only optimization: replaced out2/position_out; "
            "fine deployment is isolated in " + args.fine_deployment_ckpt,
            flush=True,
        )
    if args.freeze_layer_mixture:
        layer_weights = getattr(model.frontend, "layer_weights", None)
        if layer_weights is None:
            raise ValueError("--freeze-layer-mixture requires a learned layer mixture")
        layer_weights.requires_grad_(False)
        print("froze checkpoint layer-mixture coefficients", flush=True)
    if args.adapter_only:
        if not args.freeze or args.adapter_dim <= 0:
            raise ValueError("--adapter-only requires --freeze and --adapter-dim > 0")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("frontend.adapters."))
        adapter_names = [name for name, parameter in model.named_parameters()
                         if parameter.requires_grad]
        if not adapter_names:
            raise ValueError("--adapter-only selected no trainable adapter parameters")
        if getattr(model, "rnn", None) is not None:
            model.rnn.dropout = 0.0
        print(f"adapter-only optimization: {len(adapter_names)} tensors", flush=True)
    selected_scope: tuple[str, ...] = ()
    selected_scope_name = ""
    scope_modes = sum(bool(value) for value in (
        args.adapter_only,
        args.anchor_second_pass_only,
        args.anchor_prepass_only,
        args.encoder_only,
    ))
    if scope_modes > 1:
        raise ValueError(
            "adapter/anchor-pass/encoder-only modes are mutually exclusive"
        )
    if args.anchor_second_pass_only:
        if args.adapter_only:
            raise ValueError("--anchor-second-pass-only and --adapter-only are mutually exclusive")
        if args.arch not in ("anchor", "position_anchor"):
            raise ValueError("--anchor-second-pass-only requires an anchor architecture")
        if not args.freeze or not args.init_ckpt or not args.freeze_layer_mixture:
            raise ValueError(
                "--anchor-second-pass-only requires --freeze, --init-ckpt, and "
                "--freeze-layer-mixture"
            )
        if args.adapter_dim:
            raise ValueError("--anchor-second-pass-only requires --adapter-dim 0")
        if args.arch == "position_anchor" and args.position_weight != 0:
            raise ValueError(
                "--anchor-second-pass-only with position_anchor requires --position-weight 0"
            )
        selected_scope = select_anchor_second_pass_parameters(model)
        selected_count = sum(dict(model.named_parameters())[name].numel()
                             for name in selected_scope)
        if selected_count != ANCHOR_SECOND_PASS_PARAMETER_COUNT:
            raise ValueError(
                "anchor-second-pass trainable count mismatch: "
                f"expected {ANCHOR_SECOND_PASS_PARAMETER_COUNT:,}, got {selected_count:,}"
            )
        print(
            f"anchor-second-pass-only optimization: {len(selected_scope)} tensors / "
            f"{selected_count:,} parameters",
            flush=True,
        )
        selected_scope_name = "anchor-second-pass"
    if args.anchor_prepass_only:
        if args.adapter_only:
            raise ValueError("--anchor-prepass-only and --adapter-only are mutually exclusive")
        if args.arch not in ("anchor", "position_anchor"):
            raise ValueError("--anchor-prepass-only requires an anchor architecture")
        if not args.freeze or not args.init_ckpt or not args.freeze_layer_mixture:
            raise ValueError(
                "--anchor-prepass-only requires --freeze, --init-ckpt, and "
                "--freeze-layer-mixture"
            )
        if args.adapter_dim:
            raise ValueError("--anchor-prepass-only requires --adapter-dim 0")
        if args.arch == "position_anchor" and args.position_weight != 0:
            raise ValueError(
                "--anchor-prepass-only with position_anchor requires --position-weight 0"
            )
        selected_scope = select_anchor_prepass_parameters(model)
        selected_count = sum(dict(model.named_parameters())[name].numel()
                             for name in selected_scope)
        if selected_count != ANCHOR_PREPASS_PARAMETER_COUNT:
            raise ValueError(
                "anchor-prepass trainable count mismatch: "
                f"expected {ANCHOR_PREPASS_PARAMETER_COUNT:,}, got {selected_count:,}"
            )
        print(
            f"anchor-prepass-only optimization: {len(selected_scope)} tensors / "
            f"{selected_count:,} parameters",
            flush=True,
        )
        selected_scope_name = "anchor-prepass"
    if args.encoder_only:
        if args.arch != "position_anchor":
            raise ValueError("--encoder-only requires --arch position_anchor")
        if not args.freeze or not args.init_ckpt or not args.freeze_layer_mixture:
            raise ValueError(
                "--encoder-only requires --freeze, --init-ckpt, and "
                "--freeze-layer-mixture"
            )
        if args.adapter_dim:
            raise ValueError("--encoder-only requires --adapter-dim 0")
        if args.position_weight != 0 or args.multires_weight != 0:
            raise ValueError(
                "--encoder-only requires --position-weight 0 and --multires-weight 0"
            )
        selected_scope = select_encoder_only_parameters(model)
        for module in model.modules():
            if isinstance(module, nn.RNNBase):
                module.dropout = 0.0
        selected_count = sum(
            dict(model.named_parameters())[name].numel() for name in selected_scope
        )
        print(
            f"encoder-only optimization: {len(selected_scope)} tensors / "
            f"{selected_count:,} parameters | top {args.unfreeze_top} blocks",
            flush=True,
        )
        selected_scope_name = "encoder-only"

    if args.audit_trainable_gradients:
        if selected_scope:
            raise ValueError(
                "--audit-trainable-gradients is for joint training, not an existing scoped mode"
            )
        # With zero position weight this inference-inert head is outside the
        # objective.  Freeze it explicitly so the audited set equals the set
        # that can actually receive gradients.
        if args.arch in ("position_anchor", "native_pooled") and args.position_weight == 0:
            for name, parameter in model.named_parameters():
                if name.startswith("position_out."):
                    parameter.requires_grad_(False)
        selected_scope = tuple(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not selected_scope:
            raise ValueError("joint gradient audit selected no trainable tensors")
        selected_scope_name = "joint loss-reachable"
        print(
            f"joint gradient audit armed for {len(selected_scope)} tensors",
            flush=True,
        )

    # A treatment may initialize extra parameters (for example position_out).
    # Reset every sampling/dropout RNG after construction so matched controls see
    # the same crops, DataLoader order, and stochastic masks.
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trained_encoder_keys = sorted(
        name for name, parameter in model.named_parameters()
        if name.startswith("frontend.model.") and parameter.requires_grad
    )
    print(f"train {len(tr_items)} | val {len(va_items)} | trainable {n_tr:,} | {model_args}", flush=True)
    if trained_encoder_keys:
        print(
            f"selective SSL tuning: {len(trained_encoder_keys)} tensors / "
            f"{n_tr:,} parameters | first {trained_encoder_keys[0]} | "
            f"last {trained_encoder_keys[-1]}",
            flush=True,
        )
    initial_native_pool_fp16_sha256 = (
        fp16_tensor_fingerprint([
            (name, parameter) for name, parameter in model.named_parameters()
            if name.startswith("native_pool.")
        ])
        if args.native_only else None
    )

    parameter_groups = optimizer_parameter_groups(
        model, args.lr_ssl, args.lr_head, args.lr_layer_mix
    )
    opt = torch.optim.AdamW(parameter_groups,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps_per_epoch = len(tl) // args.accum
    total = steps_per_epoch * schedule_epochs
    warm = int(total * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
    scaler = torch.cuda.amp.GradScaler(init_scale=args.grad_scale_init)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    initialization_checkpoint_sha256 = (
        sha256_file(args.init_ckpt) if args.init_ckpt else None
    )
    fine_deployment_checkpoint_sha256 = (
        sha256_file(Path(__file__).parent / "ckpt" / args.fine_deployment_ckpt)
        if args.native_only else None
    )
    ema = None
    if args.ema > 0:
        ema = {
            key: value.detach().clone().float()
            for key, value in compact_state_dict(model, args.freeze).items()
        }
    best, hist, gstep = float("inf"), [], 0
    native_initial_authority = None
    print(f"cosine horizon {schedule_epochs} epoch(s); training for {args.epochs} epoch(s)",
          flush=True)
    eval_points = max(1, int(round(1 / args.eval_every)))
    scoped_gradients_checked = False
    for ep in range(1, args.epochs + 1):
        enter_train_mode(model, args.adapter_only, args.encoder_only)
        t0, run, run_position, run_oc_aux, seen = time.time(), 0.0, 0.0, 0.0, 0
        run_native_weighted = 0.0
        run_pre_native = 0.0
        native_bonafide_blocks = 0
        native_spoof_blocks = 0
        native_loss_accounting_max_abs_error = 0.0
        n_batches = len(tl)
        marks = {int(n_batches * k / eval_points) for k in range(1, eval_points + 1)}
        for step, (wav, lengths, lab, mask, nsegs, _) in enumerate(tl, 1):
            wav, lab = wav.to(device, non_blocking=True), lab.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(wav, lengths, nsegs)
            if args.native_only:
                # The coarse classifier replaces the final fine classifier.
                # Build the native loss below after the forward has populated
                # its exact block scores; no frame objective is mixed in.
                loss = out.float().sum() * 0.0
            elif args.arch == "crf":
                # sequence NLL (structured) + marginal CE (keeps the per-segment read-out calibrated)
                loss = model.nll(lab) + criterion(out.float().reshape(-1, 2), lab.reshape(-1))
            else:
                loss = segment_loss(out, lab, args, criterion)
            if not args.native_only and args.multires_weight > 0:
                loss = loss + args.multires_weight * multires_loss(out, lab, args.multires_tau)
            if (not args.native_only
                    and args.arch in ("span", "refinespan", "anchorspan", "multibranch")):
                ce_f, bce_b = model.aux_losses(lab)
                loss = loss + args.span_frame_weight * ce_f + args.span_boundary_weight * bce_b
            if (not args.native_only
                    and args.arch in ("refine", "anchor", "position_anchor", "native_pooled", "dynanchor", "dualanchor", "ocanchor")):
                loss = loss + 0.5 * model.aux_loss(lab)
            if not args.native_only and args.oc_aux_weight > 0:
                oc_aux_weighted = args.oc_aux_weight * angular_one_class_loss(
                    model._aux["y2"], model.out2.weight, lab,
                    alpha=args.oc_alpha, bona_margin=args.oc_m0,
                    spoof_margin=args.oc_m1,
                )
                loss = loss + oc_aux_weighted
                run_oc_aux += float(oc_aux_weighted.detach())
            if (not args.native_only
                    and args.arch in ("position_anchor", "native_pooled")):
                batch_index = (ep - 1) * n_batches + step
                position_weight = scheduled_position_weight(
                    args.position_weight, args.position_stop_frac,
                    batch_index, schedule_epochs * n_batches,
                )
                if position_weight:
                    position_weighted = position_weight * model.position_loss(lab)
                    loss = loss + position_weighted
                    run_position += float(position_weighted.detach())
            if args.arch == "native_pooled":
                native_raw, n_bonafide, n_spoof = native_bce_loss(model, lab, 160)
                native_weighted = args.native_weight * native_raw
                pre_native = float(loss.detach())
                loss = native_weighted if args.native_only else loss + native_weighted
                post_native = float(loss.detach())
                accounting_error = abs(
                    post_native
                    - (pre_native + float(native_weighted.detach()))
                )
                accounting_tolerance = max(1e-7, 1e-6 * abs(post_native))
                if not math.isfinite(post_native) or accounting_error > accounting_tolerance:
                    raise RuntimeError(
                        "native loss accounting failed: "
                        f"pre={pre_native:.9f}, weighted={float(native_weighted.detach()):.9f}, "
                        f"post={post_native:.9f}, error={accounting_error:.3e}"
                    )
                native_loss_accounting_max_abs_error = max(
                    native_loss_accounting_max_abs_error, accounting_error
                )
                run_pre_native += pre_native
                run_native_weighted += float(native_weighted.detach())
                native_bonafide_blocks += n_bonafide
                native_spoof_blocks += n_spoof
                if step == 1:
                    observed_share = float(native_weighted.detach()) / (
                        pre_native + float(native_weighted.detach())
                    )
                    if native_initial_authority is None:
                        native_initial_authority = {
                            "raw_loss": float(native_raw.detach()),
                            "weighted_loss": float(native_weighted.detach()),
                            "pre_native_loss": pre_native,
                            "observed_total_share": observed_share,
                            "bonafide_blocks": n_bonafide,
                            "spoof_blocks": n_spoof,
                        }
                        if not (args.native_authority_min <= observed_share
                                <= args.native_authority_max):
                            raise RuntimeError(
                                "native initial loss authority outside preregistered bounds: "
                                f"share={observed_share:.6%}, expected "
                                f"[{args.native_authority_min:.6%}, "
                                f"{args.native_authority_max:.6%}]"
                            )
                    print(
                        "native-160 authority: "
                        f"raw={float(native_raw.detach()):.9f} "
                        f"weighted={float(native_weighted.detach()):.9f} "
                        f"pre_native={pre_native:.9f} "
                        f"observed_total_share={observed_share:.6%} "
                        f"blocks={n_bonafide}/{n_spoof}",
                        flush=True,
                    )
            if args.arch == "boundary_unet":
                loss = loss + args.unet_boundary_weight * model.boundary_loss(lab)
            loss = loss / args.accum
            scaler.scale(loss).backward()
            if step % args.accum == 0:
                scaler.unscale_(opt)
                if selected_scope and not scoped_gradients_checked:
                    validate_selected_gradients(model, selected_scope, selected_scope_name)
                    scoped_gradients_checked = True
                    print("validated finite nonzero gradients for every scoped tensor", flush=True)
                if args.probe_only:
                    print(
                        "probe-only complete: zero optimizer steps and zero checkpoint writes",
                        flush=True,
                    )
                    return
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); gstep += 1
                if ema is not None:
                    with torch.no_grad():
                        sd_now = model.state_dict()
                        for k in ema:
                            v = sd_now[k]
                            if v.is_floating_point():
                                ema[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
                            else:
                                ema[k] = v.clone()
            run += loss.item() * args.accum; seen += 1
            if step % 200 == 0:
                position_text = (f" pos {run_position/seen:.4f}"
                                 if args.arch in ("position_anchor", "native_pooled") else "")
                oc_aux_text = (f" oc_aux {run_oc_aux/seen:.6f}"
                               if args.oc_aux_weight > 0 else "")
                native_text = ""
                if args.arch == "native_pooled":
                    native_share = run_native_weighted / max(
                        run_pre_native + run_native_weighted, 1e-12
                    )
                    native_text = (
                        f" native {run_native_weighted/seen:.6f}"
                        f" share {native_share:.3%}"
                    )
                print(f"  ep{ep} step {step}/{n_batches} loss {run/seen:.4f} "
                      f"lr {sched.get_last_lr()[-1]:.2e}{position_text}{oc_aux_text}{native_text} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step in marks:
                vloss, veer, vnative = evaluate(model, vl, device, criterion)
                if ema is not None:
                    # evaluate and save the EMA copy without disturbing the live weights
                    live = {k: v.detach().clone() for k, v in model.state_dict().items() if k in ema}
                    model.load_state_dict({k: ema[k].to(live[k].dtype) for k in ema}, strict=False)
                    _, veer_ema, _ = evaluate(model, vl, device, criterion)
                    sd_ema = {k: (ema[k].half() if ema[k].is_floating_point() else ema[k]) for k in ema}
                    torch.save({"arch": arch_name, "model_args": saved_args,
                                "model": sd_ema, "partial": bool(args.freeze), "epoch": ep - 1 + step / n_batches,
                                "val_seg_eer": veer_ema, "ema": args.ema, "args": vars(args),
                                "trained_encoder_keys": trained_encoder_keys,
                                "scoped_gradients_checked": scoped_gradients_checked},
                               str(out_path).replace(".pt", f".ep{ep - 1 + step / n_batches:g}.ema.pt"))
                    model.load_state_dict(live, strict=False)
                    print(f"  EMA({args.ema}) val segEER {veer_ema:.2f}%", flush=True)
                heer = None
                if hl is not None:
                    _, heer, _ = evaluate(model, hl, device, criterion)
                enter_train_mode(model, args.adapter_only, args.encoder_only)
                frac = ep - 1 + step / n_batches
                crit = vnative[160] if args.native_only else veer
                hist.append({"epoch": round(frac, 3), "train_loss": run / max(seen, 1),
                             "layer_mixture": layer_mixture_stats(model),
                             "position_loss_weighted": (run_position / max(seen, 1)
                                                        if args.arch in ("position_anchor", "native_pooled") else None),
                             "angular_oc_aux_weighted": (run_oc_aux / max(seen, 1)
                                                         if args.oc_aux_weight > 0 else None),
                             "native_aux_weighted": (run_native_weighted / max(seen, 1)
                                                     if args.arch == "native_pooled" else None),
                             "native_aux_observed_total_share": (
                                 run_native_weighted / max(
                                     run_pre_native + run_native_weighted, 1e-12
                                 ) if args.arch == "native_pooled" else None
                             ),
                             "native_bonafide_blocks": (native_bonafide_blocks
                                                         if args.arch == "native_pooled" else None),
                             "native_spoof_blocks": (native_spoof_blocks
                                                      if args.arch == "native_pooled" else None),
                             "val_native_eer": vnative or None,
                             "val_loss": vloss, "val_seg_eer": veer, "lps_holdout_seg_eer": heer,
                             "criterion": crit})
                extra = f" | lps-0b holdout segEER {heer:.2f}%" if heer is not None else ""
                native_extra = "".join(
                    f" | val native-{resolution_ms} EER {eer:.2f}%"
                    for resolution_ms, eer in sorted(vnative.items())
                )
                mixture = layer_mixture_stats(model)
                mixture_extra = (
                    " | mix effective "
                    f"{mixture['effective_layers']:.2f} spread "
                    f"{mixture['logit_spread']:.3f}"
                    if mixture is not None else ""
                )
                print(f"epoch {frac:.2f}: train {run/max(seen,1):.4f} | val {vloss:.4f} | "
                      f"val segEER {veer:.2f}%{native_extra}{mixture_extra}{extra} | "
                      f"{time.time()-t0:.0f}s", flush=True)
                # Save every eval point. Frozen SSL tensors are omitted, but
                # selectively adapted transformer tensors are mandatory payload.
                # graded score peaks at epoch 1 while dev keeps improving (6b7f6531 vs 647a0ae4),
                # so "best dev" is the wrong file to keep. `out_path` still tracks best dev.
                sd = {
                    key: (value.half() if value.is_floating_point() else value)
                    for key, value in compact_state_dict(model, args.freeze).items()
                }
                deployment = ({
                    "mode": "resolution_specific",
                    "fine_checkpoint": args.fine_deployment_ckpt,
                    "fine_checkpoint_sha256": fine_deployment_checkpoint_sha256,
                    "native_resolutions_ms": [160],
                    "native_model_supplies_mandatory_fine": False,
                } if args.native_only else None)
                native_pool_fp16_sha256 = (
                    fp16_tensor_fingerprint([
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if name.startswith("native_pool.")
                    ])
                    if args.native_only else None
                )
                blob = {"arch": arch_name, "model_args": saved_args,
                        "model": sd, "partial": bool(args.freeze), "epoch": frac, "val_seg_eer": veer,
                        "val_native_eer": vnative or None,
                        "layer_mixture_stats": mixture,
                        "lps_holdout_seg_eer": heer, "args": vars(args),
                        "training_objective": ("native_160_only"
                                               if args.native_only else "fine_plus_auxiliaries"),
                        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
                        "deployment": deployment,
                        "trained_encoder_keys": trained_encoder_keys,
                        "audited_trainable_keys": (list(selected_scope)
                                                   if args.audit_trainable_gradients
                                                   else None),
                        "replaced_classifier_keys": (list(replaced)
                                                     if args.native_only else None),
                        "initial_native_pool_fp16_sha256": initial_native_pool_fp16_sha256,
                        "native_pool_fp16_sha256": native_pool_fp16_sha256,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "scoped_gradients_checked": scoped_gradients_checked,
                        "native_aux_stats": ({
                            "resolution_ms": 160,
                            "raw_coefficient": args.native_weight,
                            "weighted_mean": run_native_weighted / max(seen, 1),
                            "observed_total_share": run_native_weighted / max(
                                run_pre_native + run_native_weighted, 1e-12
                            ),
                            "bonafide_blocks": native_bonafide_blocks,
                            "spoof_blocks": native_spoof_blocks,
                            "every_batch_had_both_classes": True,
                            "loss_accounting_max_abs_error": (
                                native_loss_accounting_max_abs_error
                            ),
                            "crop_alignment": args.crop_alignment,
                            "initial_authority": native_initial_authority,
                        } if args.arch == "native_pooled" else None)}
                torch.save(blob, str(out_path).replace(".pt", f".ep{frac:g}.pt"))
                if crit < best:
                    best = crit
                    torch.save(blob, out_path)
                    print(f"  saved {out_path} (criterion {crit:.2f}%)", flush=True)
                Path(str(out_path) + ".history.json").write_text(json.dumps(hist, indent=2))
    criterion_name = "native-160 EER" if args.native_only else "segment EER"
    print(f"done. best val {criterion_name} {best:.2f}% -> {out_path}")


if __name__ == "__main__":
    main()
