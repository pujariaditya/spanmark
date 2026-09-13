#!/usr/bin/env bash
# One-time setup: write .env with the paths spanmark reads, and check the
# environment can actually run the model.
#
#   ./setup.sh                         # defaults, repo-relative
#   SPANMARK_CORPUS_ROOT=/data/PS ./setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CORPUS="${SPANMARK_CORPUS_ROOT:-$ROOT/data/PartialSpoof}"
CKPT="${SPANMARK_CKPT_DIR:-$ROOT/checkpoints}"
LABELS="${SPANMARK_TASKDATA:-$ROOT/data/labels}"
RESYNTH="${SPANMARK_RESYNTH_ROOT:-$ROOT/data/resynth}"

cat > "$ROOT/.env" <<ENV
# Written by setup.sh. Source it, or export these yourself.
export SPANMARK_CORPUS_ROOT="$CORPUS"
export SPANMARK_CKPT_DIR="$CKPT"
export SPANMARK_TASKDATA="$LABELS"
export SPANMARK_RESYNTH_ROOT="$RESYNTH"
export SPANMARK_MANIFEST_DIR="$ROOT/assets/manifests"
ENV
echo "wrote $ROOT/.env"
sed 's/^/  /' "$ROOT/.env"

echo
echo "checking the environment"
python - <<'PY'
import importlib, sys
missing = []
for mod in ("numpy", "torch", "torchaudio", "soundfile", "transformers", "scipy"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:<13} {getattr(m, '__version__', '?')}")
    except Exception:
        missing.append(mod); print(f"  MISS {mod}")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else ''}")
except Exception:
    pass
if missing:
    print("\ninstall the missing packages:  pip install -e .")
    sys.exit(1)
PY

echo
mkdir -p "$CKPT" "$LABELS"
if [ ! -f "$CKPT/active.txt" ]; then
  echo "next: python scripts/download_weights.py    # fetches both routes, sha256-verified"
else
  echo "checkpoints present: $(cat "$CKPT/active.txt")"
fi
