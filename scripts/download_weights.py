#!/usr/bin/env python3
"""Fetch the released checkpoints and verify them by sha256.

    python scripts/download_weights.py          # both routes (default)
    python scripts/download_weights.py --route native

Two checkpoints are published, and both are required: the deployment is a pair,
not a single model.

  fine20      the 20 ms model. Scores every 20 ms segment. On its own it is a
              complete localizer at the fine resolution.
  native160   the released model. Predicts 160 ms blocks natively, and its
              `deployment` block names fine20 by sha256 and refuses to run
              against anything else. This is the artifact of record.

The hashes are compiled in rather than fetched alongside the files. A digest
served from the same place as the file it describes attests to nothing.

After a successful fetch this writes `active.txt` into the checkpoint directory,
which is the single selector `spanmark.predict` reads. Point SPANMARK_CKPT_DIR
somewhere else if you do not want the repo-local default.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import sys

REPO = os.environ.get("SPANMARK_HF_REPO", "RootAccess4Life/spanmark")

DEFAULT_DIR = pathlib.Path(
    os.environ.get("SPANMARK_CKPT_DIR",
                   str(pathlib.Path(__file__).resolve().parent.parent / "checkpoints"))
)

CHECKPOINTS = {
    "native": {
        "path": "native160/native160_atomic_noise_control_ep3.pt.gz",
        "sha256": "9f3a6948df5c2aeb850e3a9135867d4ed03f977d51ebcae8416e1bc9a0101c2d",
        "bytes": 313_918_955,
        "sidecar": "native160/native160_atomic_noise_control_ep3.json",
        "note": "released model -- native 160 ms head, the artifact of record",
        "active": True,
    },
    "fine": {
        "path": "fine20/xlsr_anchor_full24_resynth_ep2.pt.gz",
        "sha256": "26294df7d61823a9e3f72c4f0e896f636c2031950040a54ef7cc91903c8390d5",
        "bytes": 291_376_311,
        "sidecar": "fine20/xlsr_anchor_full24_resynth_ep2.json",
        "note": "20 ms model -- the fine partner native160 verifies by hash",
        "active": False,
    },
}


def sha256_of(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fetch(key: str, spec: dict, out_dir: pathlib.Path) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    print(f"|- {key}: {spec['note']}")
    local = hf_hub_download(repo_id=REPO, filename=spec["path"])
    actual = sha256_of(pathlib.Path(local))

    expected = spec["sha256"]
    if actual != expected:
        raise SystemExit(
            f"sha256 mismatch for {spec['path']}\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            f"Refusing to install. Do not use this file."
        )
    print(f"|   sha256 OK  {actual[:16]}...")

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / pathlib.Path(spec["path"]).name
    shutil.copyfile(local, dest)

    # The sidecar is the provenance record: recipe, matched controls, the
    # source_lock hashes, and the frozen-tensor digests. Fetch it too.
    try:
        side = hf_hub_download(repo_id=REPO, filename=spec["sidecar"])
        shutil.copyfile(side, out_dir / pathlib.Path(spec["sidecar"]).name)
    except Exception as exc:                       # provenance is not fatal
        print(f"|   (sidecar unavailable: {exc})")

    print(f"|   -> {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=("both", "native", "fine"), default="both")
    ap.add_argument("--dir", type=pathlib.Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    wanted = ("native", "fine") if args.route == "both" else (args.route,)
    if args.route != "both":
        print("note: the two-route deployment needs BOTH checkpoints; "
              "native160 will refuse to run without its fine partner.\n")

    for key in wanted:
        fetch(key, CHECKPOINTS[key], args.dir)

    if "native" in wanted:
        name = pathlib.Path(CHECKPOINTS["native"]["path"]).name
        (args.dir / "active.txt").write_text(name + "\n")
        print(f"|- active.txt -> {name}")

    print(f"\nDone. Checkpoints in {args.dir}")
    print("Set SPANMARK_CKPT_DIR to this path, or leave it unset to use the default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
