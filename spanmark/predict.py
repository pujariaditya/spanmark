#!/usr/bin/env python
"""Graded entrypoint. THIS IS THE CONTRACT -- keep the interface.

    python predict.py --dataset {ps_eval,lps} --out <path.npz>

Emit ONE SCORE PER 20 ms SEGMENT via coral_io.save_scores({utt_id: array}),
HIGHER = MORE BONAFIDE (logit(bonafide) - logit(spoof), so 0 is the boundary).

Take the segment count from column 2 of manifests/eval_<ds>.tsv. Do NOT derive
it from the audio and do NOT resample your output -- a one-segment shift keeps
the EER plausible while destroying the localization it claims to measure.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import os
import sys
from pathlib import Path

# Length-bucketed batches produce many distinct tensor shapes; without this the caching
# allocator fragments and inference on the long LlamaPartialSpoof utterances reserved ~29 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torch.utils.data import DataLoader

from spanmark.runtime.score_io import save_scores
from spanmark.data import SegmentDataset, collate, read_manifest
from spanmark.models import build_model

# Everything that points outside the package is an environment variable with a
# repo-relative default, so a clone works without editing source. `setup.sh`
# writes these into .env; `scripts/download_weights.py` populates SPANMARK_CKPT_DIR.
_REPO = Path(__file__).resolve().parent.parent

DEFAULT_EVAL_ROOT = os.environ.get("SPANMARK_EVAL_ROOT",
                                   os.environ.get("PSLPS_EVAL_ROOT", str(_REPO / "data" / "eval")))
CKPT_ROOT = Path(os.environ.get("SPANMARK_CKPT_DIR", str(_REPO / "checkpoints")))

# The active pointer is deliberately the only deployment selector; paired tune
# and full grades therefore exercise the exact same path for every endpoint.
# The pointed checkpoint's sidecar records its fixed endpoint and provenance.


def active_checkpoint() -> Path:
    """Resolve the staged checkpoint without hiding arbitrary path traversal."""
    pointer = CKPT_ROOT / "active.txt"
    if not pointer.exists():
        raise SystemExit(
            f"no {pointer} — run `python scripts/download_weights.py` first, or set "
            f"SPANMARK_CKPT_DIR to a directory containing active.txt")
    name = pointer.read_text().strip()
    if not name or Path(name).name != name:
        raise RuntimeError(f"invalid checkpoint name in {pointer}: {name!r}")
    return CKPT_ROOT / name


DATASETS = ("ps_eval", "lps")


def load_checkpoint_blob(ckpt: Path) -> dict:
    """Load a regular torch checkpoint or its deterministic gzip wrapper."""
    if ckpt.suffix == ".gz":
        with gzip.open(ckpt, "rb") as stream:
            # Torch's zip reader seeks repeatedly. On a gzip stream each
            # backwards seek re-decompresses data; buffer the exact payload once.
            payload = stream.read()
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolution_specific_deployment(blob: dict) -> tuple[Path, tuple[int, ...]] | None:
    """Resolve a fail-closed fine/native deployment declared by a checkpoint."""
    deployment = blob.get("deployment")
    if deployment is None:
        return None
    if not isinstance(deployment, dict) or deployment.get("mode") != "resolution_specific":
        raise ValueError("unsupported checkpoint deployment metadata")
    if deployment.get("native_model_supplies_mandatory_fine") is not False:
        raise ValueError("resolution-specific deployment must isolate the mandatory fine stream")
    name = deployment.get("fine_checkpoint")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"invalid fine deployment checkpoint: {name!r}")
    fine = CKPT_ROOT / name
    if not fine.is_file():
        raise ValueError(f"fine deployment checkpoint not found: {fine}")
    expected_sha = deployment.get("fine_checkpoint_sha256")
    if not isinstance(expected_sha, str) or sha256_file(fine) != expected_sha:
        raise ValueError("fine deployment checkpoint SHA-256 mismatch")
    resolutions = tuple(int(ms) for ms in deployment.get("native_resolutions_ms", ()))
    if resolutions != (160,):
        raise ValueError(f"dedicated deployment must declare only native 160 ms, got {resolutions}")
    return fine, resolutions


def validate_partial_checkpoint(model, state_keys, declared_keys=None) -> None:
    """A compact checkpoint may omit the frozen base, never adapted tensors."""
    required = {
        name for name, parameter in model.named_parameters()
        if name.startswith("frontend.model.") and parameter.requires_grad
    }
    if declared_keys is not None and set(declared_keys) != required:
        missing = sorted(required - set(declared_keys))
        extra = sorted(set(declared_keys) - required)
        raise ValueError(
            "checkpoint trained-encoder metadata disagrees with reconstructed model: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    absent = sorted(required - set(state_keys))
    if absent:
        raise ValueError(f"partial checkpoint is missing trained encoder keys: {absent[:5]}")


def decode_encoder_xor(blob: dict, model) -> dict[str, torch.Tensor]:
    """Reconstruct adapted fp16 tensors losslessly against the pinned HF base."""
    state = dict(blob["model"])
    encoding = blob.get("encoder_encoding")
    encoded = blob.get("encoder_xor")
    if encoding is None and encoded is None:
        return state
    if encoding != "fp16_xor_pretrained_v1" or not isinstance(encoded, dict):
        raise ValueError(f"unsupported encoder encoding: {encoding!r}")
    declared = set(blob.get("trained_encoder_keys", ()))
    if set(encoded) != declared:
        missing = sorted(declared - set(encoded))
        extra = sorted(set(encoded) - declared)
        raise ValueError(
            "encoded encoder keys disagree with metadata: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    base_state = model.state_dict()
    for name, xor_bits in encoded.items():
        if name not in base_state:
            raise ValueError(f"encoded encoder tensor is absent from model: {name}")
        base_half = base_state[name].detach().cpu().contiguous().half()
        bits = xor_bits.detach().cpu().contiguous()
        if bits.dtype != torch.int16 or bits.shape != base_half.shape:
            raise ValueError(
                f"invalid XOR tensor {name}: dtype={bits.dtype}, shape={tuple(bits.shape)}"
            )
        state[name] = torch.bitwise_xor(
            base_half.view(torch.int16), bits
        ).view(torch.float16)
    return state


def load_model(ckpt: Path, device: str):
    blob = load_checkpoint_blob(ckpt)
    m = build_model(blob)
    # checkpoints are stored in fp16 to keep the repo small; parameters go back to fp32.
    # For FROZEN encoders only the trainable parameters are shipped ("partial": True); the
    # encoder itself is rebuilt from the pinned HF id inside build_model, so strict=False.
    try:
        encoded_state = decode_encoder_xor(blob, m)
    except ValueError as exc:
        sys.exit(str(exc))
    sd = {
        key: value.float() if value.is_floating_point() else value
        for key, value in encoded_state.items()
    }
    if blob.get("partial", False):
        try:
            validate_partial_checkpoint(m, sd, blob.get("trained_encoder_keys"))
        except ValueError as exc:
            sys.exit(str(exc))
    missing, unexpected = m.load_state_dict(sd, strict=not blob.get("partial", False))
    if unexpected:
        sys.exit(f"checkpoint has unexpected keys: {unexpected[:5]}")
    if blob.get("partial", False):
        bad = [k for k in missing if not k.startswith("frontend.model.")]
        if bad:
            sys.exit(f"partial checkpoint is missing trainable keys: {bad[:5]}")
    return m.to(device).eval()


def length_batches(items, max_segments: int, max_batch: int):
    """Group utterances by length so padding (and peak memory) stays bounded.

    Order does not matter to the grader -- every utterance is keyed by id --
    but it matters to us: one 45 s LlamaPartialSpoof utterance padded against
    seven short ones is wasted attention compute, and eight 45 s ones at once
    is an OOM on a shared card.
    """
    order = sorted(range(len(items)), key=lambda i: items[i].n_segments)
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = items[i].n_segments
        if cur and (max(cur_max, n) * (len(cur) + 1) > max_segments or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    return batches


def collect_model_outputs(
    model,
    loader,
    want: dict[str, int],
    *,
    collect_fine: bool,
    collect_native: bool,
    expected_native: tuple[int, ...] = (),
) -> tuple[dict[str, list[float]], dict[int, dict[str, list[float]]]]:
    """Collect exactly the resolution streams assigned to one model."""
    device = next(model.parameters()).device
    fine_scores: dict[str, list[float]] = {}
    native_scores: dict[int, dict[str, list[float]]] = {}
    use_amp = device.type == "cuda" and getattr(model, "frontend_name", "") == "xlsr"
    for wav, lengths, _, _, nsegs, ids in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if collect_native:
                if not hasattr(model, "score_with_native"):
                    raise RuntimeError("dedicated native model has no native scoring path")
                fine, batch_native = model.score_with_native(
                    wav.to(device, non_blocking=True), lengths, nsegs
                )
                if tuple(sorted(batch_native)) != expected_native:
                    raise RuntimeError(
                        f"native model emitted {tuple(sorted(batch_native))}, "
                        f"expected {expected_native}"
                    )
            else:
                fine = model.score(wav.to(device, non_blocking=True), lengths, nsegs)
                batch_native = {}
        for j, utt in enumerate(ids):
            if collect_fine:
                fine_scores[utt] = fine[j, : want[utt]].float().cpu().numpy()
            for ms, coarse in batch_native.items():
                factor = ms // 20
                count = -(-want[utt] // factor)
                native_scores.setdefault(ms, {})[utt] = (
                    coarse[j, :count].float().cpu().numpy()
                )
    return fine_scores, native_scores


def release_cuda_memory() -> None:
    """Release a deleted model before constructing the next resolution model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--manifests",
                    default=os.environ.get("SPANMARK_MANIFEST_DIR",
                                           str(_REPO / "assets" / "manifests")))
    ap.add_argument("--ckpt", default=None,
                    help="default: the file named by <ckpt dir>/active.txt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-segments", type=int, default=3000,
                    help="cap on padded segments per batch (batch x longest)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    # Resolved here, not at import: the package must import without weights.
    if args.ckpt is None:
        args.ckpt = str(active_checkpoint())

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}\nTrain one (python train.py) and COMMIT it.")
    manifest = Path(args.manifests) / f"eval_{args.dataset}.tsv"
    if not manifest.exists():
        sys.exit(f"manifest not found: {manifest}")

    items = read_manifest(manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        SegmentDataset(items, args.eval_root),
        batch_sampler=length_batches(items, args.max_segments, args.batch_size),
        num_workers=args.workers, collate_fn=collate, pin_memory=True,
    )
    want = {u.utt_id: u.n_segments for u in items}
    try:
        active_blob = load_checkpoint_blob(ckpt)
        deployment = resolution_specific_deployment(active_blob)
    except ValueError as exc:
        sys.exit(str(exc))
    del active_blob
    gc.collect()

    if deployment is None:
        model = load_model(ckpt, device)
        print(
            f"{args.dataset}: {len(items)} utterances | {model.frontend_name} | {device}",
            flush=True,
        )
        scores, native_scores = collect_model_outputs(
            model, loader, want,
            collect_fine=True,
            collect_native=hasattr(model, "score_with_native"),
            expected_native=tuple(
                sorted(getattr(getattr(model, "native_pool", None), "resolutions_ms", ()))
            ),
        )
        del model
        release_cuda_memory()
    else:
        fine_ckpt, native_resolutions = deployment
        fine_model = load_model(fine_ckpt, device)
        print(
            f"{args.dataset}: {len(items)} utterances | fine={fine_model.frontend_name} "
            f"({fine_ckpt.name}) | {device}", flush=True,
        )
        scores, unexpected_native = collect_model_outputs(
            fine_model, loader, want, collect_fine=True, collect_native=False
        )
        if unexpected_native:
            sys.exit("fine deployment model unexpectedly supplied a native grid")
        del fine_model
        release_cuda_memory()

        native_model = load_model(ckpt, device)
        print(
            f"{args.dataset}: native={native_model.frontend_name} ({ckpt.name}) "
            f"at {list(native_resolutions)} ms | {device}", flush=True,
        )
        unexpected_fine, native_scores = collect_model_outputs(
            native_model, loader, want,
            collect_fine=False,
            collect_native=True,
            expected_native=native_resolutions,
        )
        if unexpected_fine:
            sys.exit("native deployment model unexpectedly supplied the mandatory fine grid")
        del native_model
        release_cuda_memory()

    missing = [u.utt_id for u in items if u.utt_id not in scores]
    if missing:
        sys.exit(f"{len(missing)} utterances unscored, e.g. {missing[:5]}")
    bad = [(u, len(scores[u]), want[u]) for u in want if len(scores[u]) != want[u]]
    if bad:
        sys.exit(f"segment-count mismatch, e.g. {bad[:3]}")

    save_scores(args.out, scores, native=native_scores or None)
    native_text = (f"; native grids {sorted(native_scores)}"
                   if native_scores else "")
    print(f"wrote {len(scores)} utterances{native_text} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
