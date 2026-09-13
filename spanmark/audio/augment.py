"""Waveform-level augmentation for localization training.

Two families:

* `SpliceAug` -- label-CHANGING. Takes an offline copy-synthesized twin of the
  utterance (see scripts/resynth.py), splices 1..k random spans of it into the
  original with a short crossfade and level matching, and sets the label of
  every 20 ms segment overlapping a span to 0 (spoof). Segments already spoof
  stay spoof. This manufactures partial spoofs with artifact families
  PartialSpoof never contains (neural codec, GAN vocoder, Griffin-Lim, LPC).

* `CrossSegmentMix` -- label-CHANGING but grid-preserving. Joins a prefix from
  one PS-train utterance to a suffix from a distinct PS-train utterance at the
  same 20 ms boundary in both waveform and labels. Position targets are later
  derived from the mixed labels, so an acoustic join is not automatically
  treated as a spoof boundary.

* `ChannelAug` -- label-PRESERVING. Additive coloured noise, synthetic reverb,
  lossy codec round-trip (mp3 / opus / vorbis / mu-law via ffmpeg), band
  limiting, gain. Applied to the whole utterance so no label moves.

Everything operates on numpy float32 at 16 kHz inside DataLoader workers.
"""

from __future__ import annotations

import math
import random
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from spanmark.data import HOP, SAMPLE_RATE

# Built by `scripts/build_resynth.py`; not shipped (11 GB of derived audio).
RESYNTH_ROOT = Path(os.environ.get(
    "SPANMARK_RESYNTH_ROOT",
    str(Path(__file__).resolve().parents[2] / "data" / "resynth")))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


class SpliceAug:
    def __init__(self, methods: list[str], split: str = "train", p: float = 0.5,
                 max_spans: int = 3, min_len_s: float = 0.2, max_len_s: float = 3.0,
                 fade_ms: tuple[float, float] = (2.0, 80.0), p_full: float = 0.05,
                 root: str | Path = RESYNTH_ROOT):
        if not methods:
            raise ValueError("SpliceAug requires at least one resynthesis method")
        if not 0.0 <= p <= 1.0 or not 0.0 <= p_full <= 1.0:
            raise ValueError("splice probabilities must be in [0, 1]")
        if max_spans < 1 or min_len_s <= 0 or max_len_s < min_len_s:
            raise ValueError("invalid splice span configuration")
        self.dirs = [Path(root) / split / m for m in methods]
        missing = [d for d in self.dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(f"resynth dirs missing: {missing}")
        self.p, self.max_spans, self.p_full = p, max_spans, p_full
        self.min_len, self.max_len = int(min_len_s * SAMPLE_RATE), int(max_len_s * SAMPLE_RATE)
        self.fade_ms = fade_ms

    def twin(self, utt_id: str) -> np.ndarray:
        d = random.choice(self.dirs)
        p = d / f"{utt_id}.wav"
        if not p.exists():
            raise FileNotFoundError(f"missing resynth twin: {p}")
        twin, sample_rate = sf.read(p, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{p}: expected {SAMPLE_RATE} Hz resynth twin, got {sample_rate}"
            )
        if twin.ndim != 1:
            raise ValueError(f"{p}: expected a mono resynth twin, got shape {twin.shape}")
        return np.ascontiguousarray(twin, dtype=np.float32)

    def __call__(self, wav: np.ndarray, y: np.ndarray, utt_id: str):
        if random.random() >= self.p:
            return wav, y
        tw = self.twin(utt_id)
        if len(tw) != len(wav):
            raise ValueError(
                f"{utt_id}: source/twin length mismatch "
                f"({len(wav)} != {len(tw)})"
            )
        n = len(wav)
        if n < 2:
            raise ValueError(f"{utt_id}: waveform is too short to splice ({n} samples)")
        wav = wav.copy()
        y = y.copy()
        n_seg = min(len(y), math.ceil(n / HOP))
        if random.random() < self.p_full:
            spans = [(0, n)]
        else:
            k = random.randint(1, self.max_spans)
            spans = []
            for _ in range(k):
                max_length = min(self.max_len, n)
                min_length = min(self.min_len, max_length)
                L = random.randint(min_length, max_length)
                a = random.randint(0, n - L) if L < n else 0
                spans.append((a, a + L))
        for a, b in spans:
            fade = int(random.uniform(*self.fade_ms) * SAMPLE_RATE / 1000)
            fade = max(1, min(fade, (b - a) // 2))
            seg = tw[a:b].copy()
            # level-match the inserted span to what it replaces
            g = _rms(wav[a:b]) / _rms(seg)
            seg *= float(np.clip(g, 0.1, 10.0))
            t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            shape = random.choice(["linear", "sin", "sin2", "log", "parabola"])   # LPS fade family
            if shape == "sin":
                ramp = np.sin(0.5 * np.pi * t)
            elif shape == "sin2":
                ramp = np.sin(0.5 * np.pi * t) ** 2
            elif shape == "log":
                ramp = np.log1p(9 * t) / np.log(10)
            elif shape == "parabola":
                ramp = 1 - (1 - t) ** 2
            else:
                ramp = t
            ramp = ramp.astype(np.float32)
            w = np.ones(b - a, dtype=np.float32)
            w[:fade] = ramp; w[-fade:] = ramp[::-1]
            wav[a:b] = wav[a:b] * (1 - w) + seg * w
            s0, s1 = a // HOP, min(n_seg, math.ceil(b / HOP))
            y[s0:s1] = 0
        return wav, y


class ChannelAug:
    def __init__(self, p_noise=0.3, p_reverb=0.2, p_codec=0.3, p_band=0.2, p_gain=0.5,
                 snr_db=(5.0, 40.0)):
        self.p_noise, self.p_reverb, self.p_codec, self.p_band, self.p_gain = (
            p_noise, p_reverb, p_codec, p_band, p_gain)
        self.snr_db = snr_db
        self._effector = None

    # -- noise: white / pink / brown via 1/f^alpha shaping
    def noise(self, wav):
        alpha = random.choice([0.0, 1.0, 2.0])
        n = len(wav)
        white = np.random.standard_normal(n).astype(np.float32)
        if alpha > 0:
            spec = np.fft.rfft(white)
            f = np.arange(spec.size, dtype=np.float32); f[0] = 1.0
            spec = spec / (f ** (alpha / 2))
            white = np.fft.irfft(spec, n).astype(np.float32)
        snr = random.uniform(*self.snr_db)
        g = _rms(wav) / (_rms(white) * (10 ** (snr / 20)))
        return wav + g * white

    def reverb(self, wav):
        rt60 = random.uniform(0.15, 0.8)
        n = int(rt60 * SAMPLE_RATE)
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        ir = np.random.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
        ir[0] = 1.0; ir /= np.sqrt((ir ** 2).sum())
        out = np.convolve(wav, ir)[: len(wav)].astype(np.float32)
        return out * (_rms(wav) / _rms(out))

    def codec(self, wav):
        from torchaudio.io import AudioEffector, CodecConfig
        choice = random.choice(["mp3", "opus", "vorbis", "mulaw"])
        x = torch.from_numpy(wav)[:, None]
        try:
            if choice == "mp3":
                eff = AudioEffector(format="mp3", codec_config=CodecConfig(bit_rate=random.choice([16000, 24000, 32000, 64000])))
            elif choice == "opus":
                eff = AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=random.choice([8000, 12000, 24000, 48000])))
            elif choice == "vorbis":
                eff = AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(qscale=random.choice([0, 2, 4])))
            else:
                eff = AudioEffector(format="wav", encoder="pcm_mulaw")
            y = eff.apply(x, SAMPLE_RATE)[:, 0].numpy()
        except Exception:
            return wav
        if len(y) < len(wav):
            y = np.pad(y, (0, len(wav) - len(y)))
        return y[: len(wav)].astype(np.float32)

    def band(self, wav):
        from scipy.signal import butter, sosfilt
        kind = random.choice(["low", "high", "band"])
        if kind == "low":
            sos = butter(4, random.uniform(3000, 7000), "low", fs=SAMPLE_RATE, output="sos")
        elif kind == "high":
            sos = butter(4, random.uniform(80, 400), "high", fs=SAMPLE_RATE, output="sos")
        else:
            sos = butter(4, [random.uniform(100, 300), random.uniform(3400, 7000)], "band",
                         fs=SAMPLE_RATE, output="sos")
        return sosfilt(sos, wav).astype(np.float32)

    def __call__(self, wav):
        if random.random() < self.p_reverb:
            wav = self.reverb(wav)
        if random.random() < self.p_noise:
            wav = self.noise(wav)
        if random.random() < self.p_band:
            wav = self.band(wav)
        if random.random() < self.p_codec:
            wav = self.codec(wav)
        if random.random() < self.p_gain:
            wav = wav * (10 ** (random.uniform(-12, 6) / 20))
        peak = np.abs(wav).max()
        if peak > 0.99:
            wav = wav * (0.99 / peak)
        return np.ascontiguousarray(wav, dtype=np.float32)


class CrossSegmentMix:
    """SAL-style distinct-utterance mixing on the exact localization grid."""

    def __init__(self, items, labels: dict[str, np.ndarray], p: float = 0.2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("cross-segment mixing probability must be in [0, 1]")
        if items is None or labels is None or len(items) < 2:
            raise ValueError(
                "cross-segment mixing needs at least two labeled PS-train items"
            )
        self.items = list(items)
        self.labels = labels
        self.p = float(p)
        self.index_by_id = {
            item.utt_id: index for index, item in enumerate(self.items)
        }
        if len(self.index_by_id) != len(self.items):
            raise ValueError("cross-segment mixing requires unique utterance IDs")
        missing_labels = [
            item.utt_id for item in self.items if item.utt_id not in self.labels
        ]
        if missing_labels:
            raise ValueError(
                f"cross-segment mixing is missing PS-train labels: {missing_labels[:3]}"
            )

    @staticmethod
    def mix_at(
        wav_a: np.ndarray,
        label_a: np.ndarray,
        wav_b: np.ndarray,
        label_b: np.ndarray,
        cut_segment: int,
        n_segments: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return A-prefix + B-suffix with one shared segment-grid cut."""
        available = min(
            len(label_a),
            len(label_b),
            math.ceil(len(wav_a) / HOP),
            math.ceil(len(wav_b) / HOP),
        )
        n = available if n_segments is None else min(int(n_segments), available)
        cut = int(cut_segment)
        if n < 2 or not 1 <= cut < n:
            raise ValueError(
                "cross-segment mix needs 1 <= cut < n_segments and n >= 2"
            )
        samples = n * HOP
        source = np.pad(
            np.asarray(wav_a, dtype=np.float32)[:samples],
            (0, max(0, samples - len(wav_a))),
        )
        donor = np.pad(
            np.asarray(wav_b, dtype=np.float32)[:samples],
            (0, max(0, samples - len(wav_b))),
        )
        waveform = np.concatenate(
            (source[: cut * HOP], donor[cut * HOP : samples])
        )
        labels = np.concatenate(
            (
                np.asarray(label_a[:cut], dtype=np.int64),
                np.asarray(label_b[cut:n], dtype=np.int64),
            )
        )
        if len(waveform) != len(labels) * HOP:
            raise RuntimeError("cross-segment mix broke waveform/label grid alignment")
        return (
            np.ascontiguousarray(waveform, dtype=np.float32),
            np.ascontiguousarray(labels, dtype=np.int64),
        )

    def __call__(self, wav: np.ndarray, y: np.ndarray, it):
        if random.random() >= self.p:
            return wav, y
        current = self.index_by_id.get(it.utt_id)
        if current is None:
            raise KeyError(
                f"cross-segment source {it.utt_id!r} is not in PS-train items"
            )
        # A size-(N-1) draw followed by a skip makes self-pairing impossible
        # without an outcome-dependent retry loop.
        partner_index = random.randrange(len(self.items) - 1)
        if partner_index >= current:
            partner_index += 1
        partner = self.items[partner_index]
        if partner.utt_id == it.utt_id:
            raise RuntimeError("cross-segment donor unexpectedly equals recipient")
        wav_b, sample_rate = sf.read(
            partner.path, dtype="float32", always_2d=False
        )
        if wav_b.ndim > 1:
            wav_b = wav_b.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{partner.path}: expected {SAMPLE_RATE} Hz, got {sample_rate}"
            )
        label_b = np.asarray(self.labels[partner.utt_id], dtype=np.int64)
        n = min(
            it.n_segments,
            partner.n_segments,
            len(y),
            len(label_b),
            math.ceil(len(wav) / HOP),
            math.ceil(len(wav_b) / HOP),
        )
        if n < 2:
            return wav, y
        return self.mix_at(
            wav,
            y,
            wav_b,
            label_b,
            random.randint(1, n - 1),
            n,
        )


class Compose:
    def __init__(
        self,
        splice: SpliceAug | None,
        channel: ChannelAug | None,
        crossmix: CrossSegmentMix | None = None,
    ):
        self.splice, self.channel, self.crossmix = splice, channel, crossmix

    def __call__(self, wav, y, it):
        if self.crossmix is not None:
            wav, y = self.crossmix(wav, y, it)
        if self.splice is not None:
            wav, y = self.splice(wav, y, it.utt_id)
        if self.channel is not None:
            wav = self.channel(wav)
        return wav, y


def build(name: str, items=None, labels=None, **kw):
    """Build a named waveform treatment.

    Splice kwargs are ``splice_methods``, ``resynth_root``, ``p_splice``,
    ``max_spans``, and ``p_full``. Cross-mixing uses ``p_crossmix``. Channel
    kwargs retain their other ``p_*`` names.
    """
    parts = set(name.split("+"))
    unknown = parts - {"splice", "channel", "crossmix"}
    if unknown:
        raise ValueError(f"unknown augmentation component(s): {sorted(unknown)}")
    splice = channel = crossmix = None
    if "splice" in parts:
        splice = SpliceAug(kw.get("splice_methods", ["encodec6"]), p=kw.get("p_splice", 0.5),
                           max_spans=kw.get("max_spans", 3), p_full=kw.get("p_full", 0.05),
                           root=kw.get("resynth_root", RESYNTH_ROOT))
    if "crossmix" in parts:
        crossmix = CrossSegmentMix(
            items, labels, p=kw.get("p_crossmix", 0.2)
        )
    if "channel" in parts:
        channel = ChannelAug(**{k: v for k, v in kw.items() if k.startswith("p_") and k not in ("p_splice", "p_full", "p_crossmix")},
                             snr_db=tuple(kw.get("snr_db", (5.0, 40.0))))
    return Compose(splice, channel, crossmix)
