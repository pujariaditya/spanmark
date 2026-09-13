#!/usr/bin/env python3
"""Publish the released checkpoints to the HuggingFace hub.

    python scripts/upload_weights.py --src <dir>            # upload both routes
    python scripts/upload_weights.py --src <dir> --dry-run  # hash and plan only

`--src` is a directory holding the four files: two `.pt.gz` checkpoints and
their two `.json` sidecars. Each is hashed locally, checked against the digests
compiled into `scripts/download_weights.py`, and only then uploaded. A file whose
hash does not match what the downloader expects is never sent — publishing a
checkpoint the downloader would reject is worse than publishing nothing.

The sidecars go up alongside the weights. They are the provenance record: the
training recipe, the matched controls it was selected against, the `source_lock`
hashes of every file that produced it, and the frozen-tensor digests taken
before and after training that show nothing frozen moved.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from download_weights import CHECKPOINTS, REPO, sha256_of  # noqa: E402


def plan(src: pathlib.Path) -> list[tuple[pathlib.Path, str, str]]:
    """-> [(local file, repo path, sha256)]; raises if anything is wrong."""
    items: list[tuple[pathlib.Path, str, str]] = []
    for key, spec in CHECKPOINTS.items():
        weight = src / pathlib.Path(spec["path"]).name
        if not weight.exists():
            raise SystemExit(f"missing {weight}")
        size = weight.stat().st_size
        if size != spec["bytes"]:
            raise SystemExit(
                f"{weight.name}: {size} bytes, expected {spec['bytes']}. "
                f"Wrong file, or a truncated copy."
            )
        digest = sha256_of(weight)
        if digest != spec["sha256"]:
            raise SystemExit(
                f"{weight.name}: sha256 {digest}\n"
                f"  download_weights.py pins {spec['sha256']}\n"
                f"Uploading this would publish a file the downloader rejects."
            )
        print(f"  {key:<7} {weight.name}  {size/1e6:7.1f} MB  {digest[:16]}  OK")
        items.append((weight, spec["path"], digest))

        sidecar = src / pathlib.Path(spec["sidecar"]).name
        if sidecar.exists():
            items.append((sidecar, spec["sidecar"], sha256_of(sidecar)))
            print(f"          {sidecar.name}  {sidecar.stat().st_size/1024:7.1f} KB  (provenance)")
        else:
            print(f"          {sidecar.name}  MISSING -- publishing without provenance")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"verifying {args.src} against the digests in download_weights.py")
    items = plan(args.src)

    if args.dry_run:
        print(f"\ndry run -- would upload {len(items)} files to {args.repo}")
        for local, remote, _ in items:
            print(f"  {local.name}  ->  {remote}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    print(f"\nrepo ready: https://huggingface.co/{args.repo}")

    for local, remote, digest in items:
        print(f"  uploading {remote} ...", flush=True)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")
        print(f"    done  {digest[:16]}")

    print(f"\nPublished to https://huggingface.co/{args.repo}")
    print("Verify a clean fetch with:  python scripts/download_weights.py --dir /tmp/ckpt-check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
