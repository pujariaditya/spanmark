#!/usr/bin/env python
"""Train the localization seed on PartialSpoof train.

    python train.py --frontend mfcc --epochs 10 --out ckpt/seed.pt

Per-segment cross-entropy on the 20 ms grid, padding masked out with
ignore_index. Selection on best validation segment EER (the graded quantity),
not loss -- they disagree, and EER is what is scored.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spanmark.data import SegmentDataset, collate, load_label_npz, read_manifest
from spanmark.models import Localizer

TASKDATA = os.environ.get(
    "SPANMARK_TASKDATA",
    os.environ.get("PSLPS_TASKDATA",
                   str(pathlib.Path(__file__).resolve().parents[1] / "data" / "labels")))


def seg_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Segment EER in percent, tie-safe (matches the grader)."""
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        # Silently returning a number here is how a degenerate validation slice
        # turns into "0.00 % EER" and a false green light.
        raise ValueError(
            f"segment EER needs both classes (bonafide={n_pos}, spoof={n_neg}); "
            "your validation slice is single-class"
        )
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tps, fps = np.cumsum(y == 1), np.cumsum(y == 0)
    keep = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    far = np.r_[0.0, fps[keep] / n_neg]
    frr = np.r_[1.0, 1.0 - tps[keep] / n_pos]
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((far[i] + frr[i]) / 2 * 100)


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, S, Y = [], [], []
    for wav, lengths, lab, mask, nsegs, _ in loader:
        wav, lab, mask = wav.to(device), lab.to(device), mask.to(device)
        out = model(wav, lengths, nsegs)
        losses.append(criterion(out.reshape(-1, 2), lab.reshape(-1)).item())
        s = (out[..., 1] - out[..., 0])[mask].float().cpu().numpy()
        S.append(s); Y.append(lab[mask].cpu().numpy())
    return float(np.mean(losses)), seg_eer(np.concatenate(S), np.concatenate(Y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=str(Path(__file__).parent / "manifests"))
    ap.add_argument("--taskdata", default=TASKDATA)
    ap.add_argument("--frontend", default="mfcc", choices=("mfcc", "xlsr"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all")
    ap.add_argument("--limit-val", type=int, default=4000)
    ap.add_argument("--out", default="ckpt/seed.pt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = args.lr if args.lr is not None else (1e-3 if args.frontend == "mfcc" else 1e-4)

    man, td = Path(args.manifests), Path(args.taskdata)
    tr_items = read_manifest(man / "ps_train.tsv")
    va_items = read_manifest(man / "ps_dev.tsv")
    if args.limit_train:
        tr_items = tr_items[: args.limit_train]
    if args.limit_val:
        va_items = va_items[: args.limit_val]
    tr_lab = load_label_npz(td / "train_labels" / "ps_train.npz")
    va_lab = load_label_npz(td / "train_labels" / "ps_dev.npz")
    print(f"train {len(tr_items)} | val {len(va_items)} | {args.frontend} | lr {lr} | {device}")

    mk = lambda it, lb, sh: DataLoader(
        SegmentDataset(it, "", lb), batch_size=args.batch_size, shuffle=sh,
        num_workers=args.workers, collate_fn=collate, pin_memory=True, drop_last=sh,
        persistent_workers=args.workers > 0)
    tl, vl = mk(tr_items, tr_lab, True), mk(va_items, va_lab, False)

    model = Localizer(args.frontend, hidden=args.hidden, dropout=args.dropout).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best, hist = float("inf"), []
    for ep in range(1, args.epochs + 1):
        model.train(); t0, run, seen = time.time(), 0.0, 0
        for step, (wav, lengths, lab, mask, nsegs, _) in enumerate(tl, 1):
            wav, lab = wav.to(device, non_blocking=True), lab.to(device)
            loss = criterion(model(wav, lengths, nsegs).reshape(-1, 2), lab.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            run += loss.item(); seen += 1
            if step % 200 == 0:
                print(f"  ep{ep} step {step}/{len(tl)} loss {run/seen:.4f}", flush=True)
        sched.step()
        vloss, veer = evaluate(model, vl, device, criterion)
        hist.append({"epoch": ep, "train_loss": run / max(seen, 1),
                     "val_loss": vloss, "val_seg_eer": veer})
        print(f"epoch {ep}: train {run/max(seen,1):.4f} | val {vloss:.4f} | "
              f"val segEER {veer:.2f}% | {time.time()-t0:.0f}s", flush=True)
        if veer < best:          # select on the GRADED quantity
            best = veer
            torch.save({"model": model.state_dict(), "frontend": args.frontend,
                        "hidden": args.hidden, "dropout": args.dropout,
                        "epoch": ep, "val_seg_eer": veer, "args": vars(args)}, out_path)
            print(f"  saved {out_path} (val segEER {veer:.2f}%)", flush=True)
    Path(out_path.parent / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"done. best val segEER {best:.2f}% -> {out_path}")


if __name__ == "__main__":
    main()
