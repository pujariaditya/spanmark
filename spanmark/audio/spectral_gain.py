"""Fixed whole-utterance spectral gains for registered training caches.

The profile stream is independent of loader order and global random state.
Labels never enter this module; all transforms preserve waveform length.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch


def _config(plan: Mapping) -> Mapping:
    if not isinstance(plan, Mapping) or not isinstance(plan.get("spectral"), Mapping):
        raise ValueError("plan must contain a spectral configuration")
    config = plan["spectral"]
    expected = {
        "sample_rate": 16000, "n_fft": 1024, "win_length": 400,
        "hop_length": 160, "window": "hamming", "window_periodic": True,
        "center": True, "pad_mode": "reflect", "normalized": False,
        "onesided": True, "profile_dtype": "float64",
        "waveform_dtype": "float32", "device": "cpu",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and type(config.get(key)) is not bool):
            raise ValueError(f"unsupported spectral setting: {key}")
    knots = np.asarray(config.get("knots_normalized"), dtype=np.float64)
    if (knots.shape != (4,) or not np.isfinite(knots).all()
            or knots[0] != 0.0 or knots[-1] != 1.0 or np.any(np.diff(knots) <= 0)):
        raise ValueError("spectral knots must be four increasing finite values from zero to one")
    limits = np.asarray(config.get("gain_db_range"), dtype=np.float64)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not limits[0] < limits[1]:
        raise ValueError("gain_db_range must contain two increasing finite values")
    probability = config.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("selection probability must lie in [0, 1]")
    return config


def build_profiles(ids: Sequence[str], plan: Mapping) -> list[dict]:
    """Draw a gate and four gains for every ID, including unselected IDs."""
    config = _config(plan)
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be an ordered sequence of utterance IDs")
    ids = list(ids)
    if not ids or any(not isinstance(utt_id, str) or not utt_id for utt_id in ids):
        raise ValueError("utterance IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("utterance IDs must be unique")
    seed = plan.get("cache", {}).get("profile_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("profile_seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    low, high = config["gain_db_range"]
    profiles = []
    for utt_id in ids:
        gate = float(rng.random())
        gains = rng.uniform(low, high, size=4)
        profiles.append({"utt_id": utt_id, "gate": gate,
                         "selected": gate < config["probability"],
                         "gains_db": gains.tolist()})
    return profiles


def frequency_gain(gains_db: Sequence[float], plan: Mapping) -> torch.Tensor:
    """Interpolate dB in float64 on physical FFT bins, then return FP32 gain."""
    config = _config(plan)
    if np.iscomplexobj(gains_db):
        raise ValueError("spectral gains must be real")
    gains = np.asarray(gains_db, dtype=np.float64)
    low, high = config["gain_db_range"]
    if (gains.shape != (4,) or not np.isfinite(gains).all()
            or np.any(gains < low) or np.any(gains > high)):
        raise ValueError("four finite in-range dB gains are required")
    frequencies = np.linspace(0.0, 1.0, config["n_fft"] // 2 + 1, dtype=np.float64)
    db = np.interp(frequencies, np.asarray(config["knots_normalized"], dtype=np.float64), gains)
    amplitude = np.power(np.float64(10.0), db / np.float64(20.0)).astype(np.float32)
    if not np.isfinite(amplitude).all() or np.any(amplitude <= 0):
        raise ValueError("spectral amplitude curve must be finite and positive")
    return torch.from_numpy(amplitude)


def transform_waveform(wave: torch.Tensor, profile: Mapping, mode: str, plan: Mapping) -> torch.Tensor:
    """Apply the registered CPU STFT operator, without clipping or resampling.

    Both selected arms take the STFT/iSTFT path. Unselected inputs are cloned
    exactly, including signed zeros, and do not take a reconstruction path.
    """
    config = _config(plan)
    if (not isinstance(wave, torch.Tensor) or wave.dtype != torch.float32
            or wave.device.type != "cpu" or wave.layout != torch.strided
            or wave.ndim != 1):
        raise ValueError("waveform must be a one-dimensional CPU float32 tensor")
    if wave.numel() <= config["n_fft"] // 2:
        raise ValueError("reflect padding requires waveform length greater than 512")
    if not bool(torch.isfinite(wave).all()):
        raise ValueError("waveform contains non-finite samples")
    if mode not in ("unity", "gain"):
        raise ValueError("spectral mode must be unity or gain")
    if not isinstance(profile, Mapping) or type(profile.get("selected")) is not bool:
        raise ValueError("profile must contain a boolean selected flag")
    if not isinstance(profile.get("utt_id"), str) or not profile["utt_id"]:
        raise ValueError("profile must contain a nonempty utterance ID")
    gate = profile.get("gate")
    if (isinstance(gate, bool) or not isinstance(gate, (int, float))
            or not math.isfinite(gate) or not 0 <= gate < 1
            or profile["selected"] != (gate < config["probability"])):
        raise ValueError("profile gate and selection disagree")
    amplitude = frequency_gain(profile.get("gains_db"), plan)
    if not profile["selected"]:
        return wave.clone()

    window = torch.hamming_window(config["win_length"], periodic=config["window_periodic"],
                                  dtype=torch.float32, device="cpu")
    spectrum = torch.stft(
        wave, n_fft=config["n_fft"], hop_length=config["hop_length"],
        win_length=config["win_length"], window=window, center=config["center"],
        pad_mode=config["pad_mode"], normalized=config["normalized"],
        onesided=config["onesided"], return_complex=True,
    )
    if mode == "gain":
        spectrum = spectrum * amplitude[:, None]
    result = torch.istft(
        spectrum, n_fft=config["n_fft"], hop_length=config["hop_length"],
        win_length=config["win_length"], window=window, center=config["center"],
        normalized=config["normalized"], onesided=config["onesided"], length=wave.numel(),
        return_complex=False,
    ).contiguous()
    if result.shape != wave.shape or result.dtype != wave.dtype or not bool(torch.isfinite(result).all()):
        raise RuntimeError("spectral reconstruction changed length/dtype or produced non-finite samples")
    if mode == "unity":
        error = float((result - wave).abs().max())
        tolerance = 2e-6 * max(1.0, float(wave.abs().max()))
        if error > tolerance:
            raise RuntimeError(f"unity roundtrip error {error} exceeds {tolerance}")
    return result
