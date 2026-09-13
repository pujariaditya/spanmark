#!/usr/bin/env bash
# Reproduce the reported segment EER.
#
# Needs an evaluation manifest built by scripts/prepare_data.py and labels under
# SPANMARK_TASKDATA. Reports EER at every resolution on the grid; the headline
# number is the 160 ms cell.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
cd "$ROOT"

MANIFEST="${SPANMARK_EVAL_MANIFEST:-assets/manifests/ps_eval.tsv}"
OUT="${SPANMARK_EVAL_OUT:-out/ps_eval_scores.npz}"

if [ ! -f "$MANIFEST" ]; then
  echo "no $MANIFEST"
  echo "build one first:"
  echo "  python scripts/prepare_data.py --corpus-root \"\$SPANMARK_CORPUS_ROOT\" --split eval"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "scoring $(grep -vc '^#' "$MANIFEST") utterances"
python -m spanmark.predict --dataset ps_eval --out "$OUT"

echo
python -m spanmark.evaluate --scores "$OUT" --manifest "$MANIFEST"
