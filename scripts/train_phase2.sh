#!/usr/bin/env bash
# Phase 2 — the native 160 ms head.
#
# Starts from the phase-1 fine checkpoint (or the released fine20) and trains
# ONLY the interval head: 12 tensors, 34,118 parameters. Everything else — 30
# native tensors, 40 reader tensors, all 384 XLS-R encoder tensors — is frozen,
# and the run receipt records their digest before and after to prove it.
#
# This is the cheap half. It is what makes the contribution reproducible without
# rebuilding phase 1, which is why fine20 is published.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
cd "$ROOT"

CKPT="${SPANMARK_CKPT_DIR:-$ROOT/checkpoints}"
PARENT="${1:-$CKPT/xlsr_anchor_full24_resynth_ep2.pt.gz}"
OUT="${SPANMARK_OUT:-out/phase2}"

if [ ! -f "$PARENT" ]; then
  echo "no parent checkpoint at $PARENT"
  echo "fetch it:  python scripts/download_weights.py --route fine"
  exit 1
fi
mkdir -p "$OUT"

python -m spanmark.training.train_xlsr \
  --train-manifest assets/manifests/ps_train.tsv \
  --dev-manifest   assets/manifests/ps_dev.tsv \
  --init-from "$PARENT" \
  --native-only --native-resolutions 160 \
  --fine-deployment-ckpt "$(basename "$PARENT")" \
  --head interval \
  --freeze-encoder --freeze-reader \
  --augment atomic_wave_noise \
  --epochs 3 --seed 1234 \
  --out "$OUT" "${@:2}"

echo
echo "phase 2 done -> $OUT"
echo "the emitted checkpoint carries a deployment block naming its fine partner by sha256."
