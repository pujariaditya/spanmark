#!/usr/bin/env python3
"""Rebuild the resynthesis twin corpus that phase-1 training splices from.

    python scripts/build_resynth.py --split train --methods encodec6 lpc
    python scripts/build_resynth.py --split train --methods all --workers 4

For every utterance in a manifest this writes a *twin*: the same speech put
through a codec or vocoder round-trip, so it carries synthesis artifacts while
keeping the content and, critically, the timing. `SpliceAug` then cuts spans out
of a twin and pastes them into the original, relabelling the covered 20 ms
segments as spoof — an imitation of how PartialSpoof itself was built, where a
segment is swapped in from a regenerated version of the same speech.

Layout, which `spanmark.audio.augment.SpliceAug` reads directly:

    <SPANMARK_RESYNTH_ROOT>/<split>/<method>/<utt_id>.wav

**Every twin is sample-exact with its source.** That is not a nicety. The splice
copies span `[a, b)` from the twin into the original at the same indices, and
the label edit covers exactly the 20 ms segments that span touches. A twin one
sample longer would shift every label after it. Each method below therefore ends
by trimming or zero-padding to the source length, and the writer asserts it.

## Methods

  encodec6     EnCodec at ~6 kbps (facebook/encodec_24khz). Neural codec
               round-trip: 16k -> 24k -> encode -> decode -> 16k.
  lpc          Classic LPC vocoder. Order-16 analysis, then resynthesis from
               pulse/noise excitation rather than the true residual, which is
               what makes it a vocoder rather than a filter.
  griffinlim   Magnitude STFT -> Griffin-Lim phase reconstruction. Present in
               the original corpus but NOT used by the released recipe.
  hifigan      Mel -> HiFi-GAN. Needs `pip install speechbrain`; the other three
               run on the base dependencies.

The released fine checkpoint trained on a uniform choice over
**encodec6 / hifigan / lpc**.

## This will not reproduce the original bytes

The generator that produced the twins the released checkpoint trained on was not
preserved. This script reconstructs the recipe from the checkpoint sidecar, and
the recipe is faithful, but a different EnCodec build or a different HiFi-GAN
checkpoint gives different samples. Expect a model trained on these twins to
land near the released one, not on it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

SAMPLE_RATE = 16_000
ALL_METHODS = ("encodec6", "lpc", "griffinlim", "hifigan")
RECIPE_METHODS = ("encodec6", "hifigan", "lpc")   # what the release trained on


# --------------------------------------------------------------------- helpers
def _fit(y: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples. The whole contract in one line."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.concatenate([y, np.zeros(n - len(y), np.float32)])
    return y


def _match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Restore the source's loudness; a vocoder pass usually shifts it."""
    a = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    b = float(np.sqrt(np.mean(np.square(ref)) + 1e-12))
    out = y * (b / a) if a > 1e-9 else y
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return (out / peak * 0.999).astype(np.float32) if peak > 0.999 else out.astype(np.float32)


# --------------------------------------------------------------------- methods
class Encodec6:
    """Neural codec round-trip at roughly 6 kbps."""
    name = "encodec6"

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import EncodecModel, AutoProcessor
        self.torch = torch
        self.device = device
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()
        self.proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.sr = self.model.config.sampling_rate          # 24 kHz

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        import torchaudio
        torch = self.torch
        n = len(wav)
        x = torch.from_numpy(wav)[None]
        x24 = torchaudio.functional.resample(x, SAMPLE_RATE, self.sr)
        inputs = self.proc(raw_audio=x24[0].numpy(), sampling_rate=self.sr,
                           return_tensors="pt")
        with torch.inference_mode():
            enc = self.model.encode(inputs["input_values"].to(self.device),
                                    inputs["padding_mask"].to(self.device),
                                    bandwidth=6.0)
            dec = self.model.decode(enc.audio_codes, enc.audio_scales,
                                    inputs["padding_mask"].to(self.device))[0]
        y24 = dec[0, 0].detach().cpu()[None]
        y16 = torchaudio.functional.resample(y24, self.sr, SAMPLE_RATE)[0].numpy()
        return _match_rms(_fit(y16, n), wav)


class LPCVocoder:
    """Order-16 LPC analysis, resynthesis from pulse/noise excitation.

    Keeping the true residual would make this an identity filter. Replacing it
    with a pulse train where the frame is voiced and noise where it is not is
    what produces a vocoder's characteristic artifacts, which is the point.
    """
    name = "lpc"

    def __init__(self, order: int = 16, frame_ms: float = 25.0, hop_ms: float = 10.0,
                 seed: int = 1234):
        self.order = order
        self.frame = int(SAMPLE_RATE * frame_ms / 1000)
        self.hop = int(SAMPLE_RATE * hop_ms / 1000)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _levinson(r: np.ndarray, order: int) -> np.ndarray:
        a = np.zeros(order + 1); a[0] = 1.0
        e = r[0]
        if e <= 0:
            return a
        for i in range(1, order + 1):
            acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else r[1]
            k = -acc / e
            a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
            e *= (1.0 - k * k)
            if e <= 0:
                break
        return a

    def _pitch(self, frame: np.ndarray) -> int:
        """Crude autocorrelation pitch; 0 means unvoiced."""
        lo, hi = SAMPLE_RATE // 400, SAMPLE_RATE // 70      # 70-400 Hz
        f = frame - frame.mean()
        if len(f) <= hi or float(np.sqrt(np.mean(f ** 2))) < 1e-4:
            return 0
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        if ac[0] <= 0:
            return 0
        seg = ac[lo:hi]
        lag = int(np.argmax(seg)) + lo
        return lag if ac[lag] / ac[0] > 0.30 else 0

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        n = len(wav)
        win = np.hanning(self.frame).astype(np.float32)
        out = np.zeros(n + self.frame, np.float32)
        norm = np.zeros(n + self.frame, np.float32)
        phase = 0

        for start in range(0, max(n - self.frame, 0) + 1, self.hop):
            seg = wav[start:start + self.frame]
            if len(seg) < self.frame:
                seg = _fit(seg, self.frame)
            w = seg * win
            ac = np.correlate(w, w, mode="full")[self.frame - 1:][: self.order + 1]
            if ac[0] <= 1e-12:
                continue
            a = self._levinson(ac.astype(np.float64), self.order)
            resid = lfilter(a, [1.0], w)
            gain = float(np.sqrt(np.mean(resid ** 2) + 1e-12))

            lag = self._pitch(seg)
            if lag:                                   # voiced: pulse train
                exc = np.zeros(self.frame, np.float32)
                idx = np.arange(phase % lag, self.frame, lag)
                exc[idx.astype(int)] = 1.0
                exc *= np.sqrt(lag)                   # keep energy comparable
                phase = int((phase + self.frame) % lag)
            else:                                     # unvoiced: white noise
                exc = self.rng.standard_normal(self.frame).astype(np.float32)
                phase = 0
            exc *= gain / (np.sqrt(np.mean(exc ** 2)) + 1e-12)

            syn = lfilter([1.0], a, exc).astype(np.float32) * win
            out[start:start + self.frame] += syn
            norm[start:start + self.frame] += win ** 2

        good = norm > 1e-6
        out[good] /= norm[good]
        return _match_rms(_fit(out, n), wav)


class GriffinLimVocoder:
    """Magnitude STFT, phase thrown away and reconstructed iteratively."""
    name = "griffinlim"

    def __init__(self, n_fft: int = 1024, hop: int = 256, iters: int = 32):
        import torch, torchaudio
        self.torch = torch
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop, power=1.0)
        self.gl = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop,
                                                   power=1.0, n_iter=iters)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        n = len(wav)
        with torch.inference_mode():
            y = self.gl(self.spec(torch.from_numpy(wav))).numpy()
        return _match_rms(_fit(y, n), wav)


class HiFiGAN:
    """Mel -> HiFi-GAN. Optional: needs speechbrain."""
    name = "hifigan"

    def __init__(self, device: str = "cpu"):
        try:
            from speechbrain.inference.vocoders import HIFIGAN
        except Exception as exc:                      # pragma: no cover
            raise SystemExit(
                "the hifigan method needs speechbrain:\n"
                "    pip install speechbrain\n"
                f"({exc})\n"
                "The other three methods run on the base dependencies. The released "
                "recipe used encodec6/hifigan/lpc, so dropping hifigan changes the "
                "augmentation distribution — say so if you report a number."
            )
        import torch, torchaudio
        self.torch, self.torchaudio = torch, torchaudio
        self.model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz",
            savedir=os.environ.get("SPANMARK_HIFIGAN_DIR", "data/hifigan"),
            run_opts={"device": device})
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80,
            f_min=0, f_max=8000, power=1.0, norm="slaney", mel_scale="slaney").to(device)
        self.device = device

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        n = len(wav)
        with torch.inference_mode():
            m = self.mel(torch.from_numpy(wav).to(self.device))
            m = torch.log(torch.clamp(m, min=1e-5))
            y = self.model.decode_batch(m[None])[0, 0].cpu().numpy()
        return _match_rms(_fit(y, n), wav)


BUILDERS = {"encodec6": Encodec6, "lpc": LPCVocoder,
            "griffinlim": GriffinLimVocoder, "hifigan": HiFiGAN}


# ------------------------------------------------------------------------ main
def read_manifest(path: pathlib.Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append((parts[0], parts[2]))
    return rows


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "dev"), default="train")
    ap.add_argument("--methods", nargs="+", default=list(RECIPE_METHODS),
                    help=f"any of {ALL_METHODS}, or 'all', or 'recipe'")
    ap.add_argument("--manifest", type=pathlib.Path)
    ap.add_argument("--corpus-root", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("SPANMARK_CORPUS_ROOT",
                                                        str(repo / "data" / "PartialSpoof"))))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("SPANMARK_RESYNTH_ROOT",
                                                        str(repo / "data" / "resynth"))))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    methods = (list(ALL_METHODS) if args.methods == ["all"]
               else list(RECIPE_METHODS) if args.methods == ["recipe"]
               else args.methods)
    bad = [m for m in methods if m not in BUILDERS]
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; choose from {ALL_METHODS}")

    manifest = args.manifest or repo / "assets" / "manifests" / f"ps_{args.split}.tsv"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest}")
    rows = read_manifest(manifest)
    if args.limit:
        rows = rows[: args.limit]

    import soundfile as sf
    print(f"{len(rows):,} utterances x {len(methods)} method(s) -> {args.out}/{args.split}")

    for method in methods:
        out_dir = args.out / args.split / method
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{method}] building into {out_dir}", flush=True)
        build = BUILDERS[method](args.device) if method in ("encodec6", "hifigan") \
            else BUILDERS[method]()
        made = skipped = failed = 0
        for i, (utt, rel) in enumerate(rows, 1):
            dest = out_dir / f"{utt}.wav"
            if dest.exists():
                skipped += 1
                continue
            src = args.corpus_root / rel
            if not src.exists():
                failed += 1
                continue
            try:
                wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav[:, 0]
                if sr != SAMPLE_RATE:
                    import torch, torchaudio
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav)[None], sr, SAMPLE_RATE)[0].numpy()
                twin = build(np.ascontiguousarray(wav, dtype=np.float32))
                assert len(twin) == len(wav), \
                    f"{utt}: twin {len(twin)} != source {len(wav)} — would shift every label"
                # Write to a sidecar then rename, so an interrupted run never
                # leaves a half-written twin that a later run would skip.
                # soundfile infers the container from the extension, and ".part"
                # is not one, so name the format explicitly.
                tmp = dest.with_suffix(".part")
                sf.write(str(tmp), twin, SAMPLE_RATE, subtype="PCM_16", format="WAV")
                tmp.replace(dest)
                made += 1
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"   {utt}: {type(exc).__name__}: {exc}", flush=True)
            if i % 500 == 0:
                print(f"   {i:,}/{len(rows):,}  made={made} skipped={skipped} failed={failed}",
                      flush=True)
        print(f"[{method}] made {made}, already present {skipped}, failed {failed}")

    print(f"\nSet SPANMARK_RESYNTH_ROOT={args.out} before training phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
