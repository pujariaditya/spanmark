"""Deterministic timestamp-preserving atoms adapted from RawBoost.

The signal-dependent impulse and stationary signal-independent noise
arithmetic follows SentryMao/SAL commit
``b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485`` and upstream RawBoost.
Only these two atoms are retained: no filter is ever applied to the source
waveform, so source samples are neither shifted nor mixed across label edges.

Copyright (c) 2026 Yuchen Mao and copyright (c) 2021 Hemlata Tak et al.;
both upstream projects distribute this code under the MIT license.
"""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
from scipy import signal


RNG_NAMESPACE = "native160-atomic-wave-noise-v1"
SAMPLE_RATE = 16_000
VALID_MODES = ("clean", "impulse", "stationary", "impulse_stationary")


def _rand_range(x1: float, x2: float, integer: bool, *, rng) -> float | int:
    """Retain RawBoost's one-element legacy RandomState draw."""
    value = rng.uniform(low=x1, high=x2, size=(1,))
    return int(value[0]) if integer else float(value[0])


def _normalise_peak_if_needed(x: np.ndarray, always: bool) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if always:
        if peak == 0:
            raise ValueError("cannot peak-normalize an all-zero noise vector")
        return x / peak
    return x / peak if peak > 1 else x


def _notch_coefficients(
    *, rng, n_bands: int = 5, min_frequency: float = 20,
    max_frequency: float = 8000, min_bandwidth: float = 100,
    max_bandwidth: float = 1000, min_coefficients: int = 10,
    max_coefficients: int = 100, sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    coefficients: np.ndarray | int = 1
    for _ in range(n_bands):
        center = _rand_range(min_frequency, max_frequency, False, rng=rng)
        bandwidth = _rand_range(min_bandwidth, max_bandwidth, False, rng=rng)
        count = _rand_range(min_coefficients, max_coefficients, True, rng=rng)
        if count % 2 == 0:
            count += 1
        low = max(center - bandwidth / 2, 0.001)
        high = min(center + bandwidth / 2, sample_rate / 2 - 0.001)
        bandstop = signal.firwin(
            count, [low, high], window="hamming", fs=sample_rate
        )
        coefficients = np.convolve(bandstop, coefficients)
    _, response = signal.freqz(coefficients, 1, fs=sample_rate)
    return np.asarray(coefficients) / np.max(np.abs(response))


def _centered_filter(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Filter and compensate the odd FIR delay without advancing labels."""
    if x.ndim != 1 or coefficients.ndim != 1 or not coefficients.size:
        raise ValueError("FIR expects nonempty one-dimensional arrays")
    if coefficients.size % 2 != 1:
        raise ValueError("the concatenated FIR must have odd length")
    delay = (coefficients.size - 1) // 2
    padded = np.pad(x, (0, coefficients.size + 1), mode="constant")
    filtered = signal.lfilter(coefficients, 1, padded)
    return filtered[delay:delay + x.size]


def signal_dependent_impulses(
    waveform: np.ndarray, *, rng, maximum_fraction_percent: float = 10,
    gain: float = 2,
) -> tuple[np.ndarray, dict]:
    """Apply RawBoost's sparse signal-dependent impulse operation."""
    fraction = _rand_range(0, maximum_fraction_percent, False, rng=rng)
    count = int(waveform.size * fraction / 100)
    positions = rng.permutation(waveform.size)[:count]
    factors = ((2 * rng.rand(count) - 1) * (2 * rng.rand(count) - 1))
    output = waveform.copy()
    output[positions] = waveform[positions] + gain * waveform[positions] * factors
    output = _normalise_peak_if_needed(output, always=False)
    return output, {
        "fraction_percent": fraction,
        "count": count,
        "gain": gain,
    }


def stationary_colored_noise(
    waveform: np.ndarray, *, rng, snr_min_db: float = 10,
    snr_max_db: float = 40,
) -> tuple[np.ndarray, dict]:
    """Add independently generated RawBoost stationary colored noise."""
    noise = rng.normal(0, 1, waveform.size)
    coefficients = _notch_coefficients(rng=rng)
    noise = _centered_filter(noise, coefficients)
    noise = _normalise_peak_if_needed(noise, always=True)
    snr_db = _rand_range(snr_min_db, snr_max_db, False, rng=rng)
    source_norm = float(np.linalg.norm(waveform, 2))
    if source_norm == 0:
        scaled = np.zeros_like(noise)
    else:
        scaled = noise / np.linalg.norm(noise, 2) * source_norm / 10 ** (0.05 * snr_db)
    return waveform + scaled, {
        "snr_db": snr_db,
        "realized_snr_db": (
            float("inf") if not np.any(scaled)
            else 20 * np.log10(source_norm / np.linalg.norm(scaled, 2))
        ),
        "filter_coefficients": int(coefficients.size),
    }


def keyed_seeds(utt_id: str, seed: int = 1234, epoch: int = 0) -> dict:
    payload = {
        "namespace": RNG_NAMESPACE, "seed": int(seed), "epoch": int(epoch),
        "utt_id": str(utt_id), "visit": 0,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).digest()
    return {
        "digest_sha256": digest.hex(),
        "python_seed": int.from_bytes(digest[:8], "big"),
        "numpy_seed": int.from_bytes(digest[8:12], "big"),
    }


def selected_for_augmentation(
    utt_id: str, probability: float, seed: int = 1234, epoch: int = 0,
) -> tuple[bool, float, dict]:
    if not 0 <= probability <= 1:
        raise ValueError("augmentation probability must be in [0, 1]")
    seeds = keyed_seeds(utt_id, seed=seed, epoch=epoch)
    gate = random.Random(seeds["python_seed"]).random()
    return gate < probability, gate, seeds


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class AtomicWaveNoise:
    """Sample-keyed atomic augmentation with exact clean replay."""

    preserves_grid = True

    def __init__(
        self, mode: str, probability: float, seed: int = 1234, epoch: int = 0,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown atomic-wave-noise mode: {mode}")
        if mode == "clean" and probability != 0:
            raise ValueError("clean mode requires probability zero")
        if mode != "clean" and probability != 0.5:
            raise ValueError("registered treatments require probability 0.5")
        self.mode = mode
        self.probability = float(probability)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def selected(self, utt_id: str) -> bool:
        return selected_for_augmentation(
            utt_id, self.probability, self.seed, self.epoch
        )[0]

    def __call__(self, waveform: np.ndarray, utt_id: str) -> tuple[np.ndarray, dict]:
        source = np.ascontiguousarray(waveform, dtype=np.float32)
        if source.ndim != 1 or not source.size or not np.isfinite(source).all():
            raise ValueError("waveform must be finite, nonempty, and mono")
        selected, gate, seeds = selected_for_augmentation(
            utt_id, self.probability, self.seed, self.epoch
        )
        result: np.ndarray = source
        stages: dict[str, dict] = {}
        if selected:
            rng = np.random.RandomState(seeds["numpy_seed"])
            if self.mode in ("impulse", "impulse_stationary"):
                result, stages["impulse"] = signal_dependent_impulses(result, rng=rng)
            if self.mode in ("stationary", "impulse_stationary"):
                result, stages["stationary"] = stationary_colored_noise(result, rng=rng)
            result = np.ascontiguousarray(result, dtype=np.float32)
        if result.shape != source.shape or not np.isfinite(result).all():
            raise RuntimeError("atomic augmentation changed length or produced nonfinite audio")
        changed = int(np.count_nonzero(source != result))
        receipt = {
            "version": 1, "mode": self.mode, "utt_id": str(utt_id),
            "probability": self.probability, "gate": gate, "selected": selected,
            "epoch": self.epoch, "rng_namespace": RNG_NAMESPACE, **seeds,
            "source_sha256": array_sha256(source),
            "output_sha256": array_sha256(result),
            "length": int(source.size), "changed_samples": changed,
            "delta_l2": float(np.linalg.norm(
                result.astype(np.float64) - source.astype(np.float64)
            )),
            "stages": stages,
        }
        if self.mode == "clean" and not np.array_equal(result, source):
            raise RuntimeError("clean atomic path changed source bytes")
        zero_count_impulse = (
            self.mode == "impulse"
            and stages.get("impulse", {}).get("count") == 0
        )
        if (
            selected and self.mode != "clean" and not changed and np.any(source)
            and not zero_count_impulse
        ):
            raise RuntimeError("selected atomic treatment was inert on non-silent audio")
        return result, receipt
