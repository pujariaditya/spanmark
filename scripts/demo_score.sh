#!/usr/bin/env bash
# Score a handful of utterances end to end and show what came out.
#
# This is the smallest thing that exercises the whole contract: both routes
# load, the deployment gate verifies the fine partner by hash, and the output
# carries BOTH grids. A missing native grid is the failure that matters — the
# scorer would silently fall back to pooling and the number would collapse with
# no error anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
cd "$ROOT"

MANIFEST="${1:-assets/manifests/ps_dev.tsv}"
N="${SPANMARK_DEMO_N:-6}"
OUT="${SPANMARK_DEMO_OUT:-out/demo_scores.npz}"
mkdir -p "$(dirname "$OUT")"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
head -1 "$MANIFEST" > "$TMP/demo.tsv"
grep -v '^#' "$MANIFEST" | head -"$N" >> "$TMP/demo.tsv"
echo "scoring $N utterances from $(basename "$MANIFEST")"

python -m spanmark.predict --dataset ps_eval --out "$OUT" \
       --manifests "$TMP" --eval-root "${SPANMARK_CORPUS_ROOT:-data/PartialSpoof}"

python - "$OUT" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1])
keys = set(z.files)
print(f"\n  arrays: {sorted(keys)}")
print(f"  20 ms : {len(z['scores']):,} segment scores over {len(z['offsets'])} utterances")
if "scores_160" in keys:
    print(f"  160 ms: {len(z['scores_160']):,} block scores   <- native grid present")
else:
    raise SystemExit("\n  FAIL: no native 160 ms grid. The coarse route did not run, and a "
                     "scorer would silently min-pool instead. Check active.txt and the "
                     "deployment block.")
s = z["scores"]
print(f"  range : {s.min():.3f} .. {s.max():.3f}   (higher = more bonafide)")
print("\n  OK — both grids emitted.")
PY
