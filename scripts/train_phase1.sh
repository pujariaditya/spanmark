#!/usr/bin/env bash
# Phase 1 — the 20 ms fine route.
#
# Adapts all 24 XLS-R blocks while freezing the inherited reader and the learned
# state mixture. The adaptation is tiny: relative L2 change 0.00136887, a 0.14 %
# nudge. That is not an accident of the schedule, it is the result — the reader
# is the anchor the encoder learned against, and unfreezing it invalidates the
# encoder's adaptation.
#
# Needs the resynthesis twin corpus (scripts/build_resynth.py). Without it,
# SpliceAug cannot run and the recipe degrades to the no-augmentation control,
# which measured 0.51 worse against a run variance of 0.2522.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
cd "$ROOT"

OUT="${SPANMARK_OUT:-out/phase1}"
mkdir -p "$OUT"

python -m spanmark.training.train_xlsr \
  --train-manifest assets/manifests/ps_train.tsv \
  --dev-manifest   assets/manifests/ps_dev.tsv \
  --frontend xlsr \
  --adapt-all-blocks \
  --freeze-reader \
  --augment splice \
  --splice-prob 0.5 \
  --splice-spans 1 3 \
  --splice-seconds 0.2 3.0 \
  --crossfade-ms 2 80 \
  --full-replacement-prob 0.05 \
  --loss focal --focal-gamma 2 \
  --pass-one-ce-weight 0.5 \
  --angular-one-class-weight 0.001 \
  --optimizer adamw --ssl-lr 1e-6 --weight-decay 0.01 \
  --epochs 2 --warmup 0 --cosine-horizon 8 \
  --batch-size 16 --crop-segments 300 --workers 8 --seed 1234 \
  --out "$OUT" "$@"

echo
echo "phase 1 done -> $OUT"
echo "phase 2 starts from this checkpoint, or from the released fine20."
