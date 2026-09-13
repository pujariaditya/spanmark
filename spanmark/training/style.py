"""Training-only mean shifts between equally labeled cached native blocks.

This module has no model parameters and is never used for inference. The full
valid frame-label pattern, length, and source utterance constrain donor choice.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
import torch


def pattern_keys(frame_labels, lengths):
    """Encode ordered valid labels plus length; padded labels never contribute."""
    labels = np.asarray(frame_labels)
    lengths = np.asarray(lengths, dtype=np.int64)
    if (labels.ndim != 2 or labels.shape[1] != 8
            or lengths.shape != (len(labels),)
            or np.any((lengths < 1) | (lengths > 8))):
        raise ValueError("mean style requires nonempty blocks of at most eight frames")
    valid = np.arange(8)[None, :] < lengths[:, None]
    if not np.isin(labels[valid], (0, 1)).all():
        raise ValueError("valid frame labels must be binary")
    bits = (np.where(valid, labels, 0).astype(np.int64)
            * (1 << np.arange(8, dtype=np.int64))[None, :]).sum(axis=1)
    return lengths * 256 + bits


def choose_donors(keys, utterances, rng):
    """Uniformly select another-utterance member of each identical-key group.

    Sorting each group by utterance makes the forbidden members contiguous.
    A uniform rank over the remainder maps past that interval, avoiding biased
    rejection limits or a preference for particular donor utterances. With
    rng=None, return a deterministic representative for a disabled-step audit.
    """
    keys, utterances = np.asarray(keys), np.asarray(utterances, dtype=np.int64)
    if keys.ndim != 1 or utterances.shape != keys.shape:
        raise ValueError("pattern keys and utterance identifiers must align")
    donors = np.full(len(keys), -1, dtype=np.int64)
    for key in np.unique(keys):
        members = np.flatnonzero(keys == key)
        ordered = members[np.argsort(utterances[members], kind="stable")]
        ordered_utterances = utterances[ordered]
        left = np.searchsorted(ordered_utterances, ordered_utterances, side="left")
        right = np.searchsorted(ordered_utterances, ordered_utterances, side="right")
        available = len(ordered) - (right - left)
        eligible = available > 0
        if eligible.any():
            ranks = (np.zeros(int(eligible.sum()), dtype=np.int64) if rng is None
                     else rng.integers(available[eligible]))
            ranks += np.where(ranks >= left[eligible], right[eligible] - left[eligible], 0)
            donors[ordered[eligible]] = ordered[ranks]
    return donors


def _empty_pattern_audit():
    return dict(blocks=0, eligible=0, requested=0, selected=0, applied=0,
                first_batch_skipped=0, shift_l2_sum=0.0, shift_l2_max=0.0)


class NativeMeanStyle:
    """One independent augmentation stream with cumulative provenance counters."""

    def __init__(self, *, seed, strength, probability):
        if not 0 <= strength <= 1 or not 0 <= probability <= 1:
            raise ValueError("strength and probability must lie in [0, 1]")
        self.strength = float(strength)
        self.probability = float(probability)
        # Namespace 2 is reserved for style; order/phase generators are separate.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.by_pattern = defaultdict(_empty_pattern_audit)
        self.batches = 0
        self._selection_hash = hashlib.sha256()
        self.last_selection = None

    def __call__(self, states, frame_labels, lengths, utterances, *, optimizer_step):
        """Shift selected valid frames once, using means of the original batch.

        The caller supplies FP32 states, as the native head itself consumes.
        Masks and labels are not mutated. Step one deliberately performs no
        random sampling or arithmetic on states, preserving loss calibration.
        """
        if states.dtype != torch.float32 or states.ndim != 3 or states.shape[1] != 8:
            raise ValueError("mean style consumes FP32 [block, 8, channel] states")
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("optimizer_step must be a positive one-based integer")
        lengths = np.asarray(lengths, dtype=np.int64)
        utterances = np.asarray(utterances, dtype=np.int64)
        keys = pattern_keys(frame_labels, lengths)
        if len(keys) != len(states) or utterances.shape != keys.shape:
            raise ValueError("states and block metadata do not align")
        self.batches += 1
        first_batch = optimizer_step == 1
        # Step one audits deterministic donor eligibility without consuming the
        # dedicated training augmentation stream.
        donors = choose_donors(keys, utterances, None if first_batch else self.rng)
        eligible = donors >= 0
        requested = (np.zeros(len(keys), dtype=bool) if first_batch
                     else self.rng.random(len(keys)) < self.probability)
        selected = eligible & requested
        applied = selected & (self.strength > 0)
        norms = np.zeros(len(keys), dtype=np.float64)
        output = states
        if applied.any():
            valid = torch.arange(8, device=states.device)[None, :] < torch.as_tensor(
                lengths, device=states.device)[:, None]
            means = states.masked_fill(~valid[..., None], 0).sum(1) / torch.as_tensor(
                lengths, device=states.device, dtype=torch.float32)[:, None]
            rows = np.flatnonzero(applied)
            source_rows = torch.as_tensor(rows, device=states.device)
            donor_rows = torch.as_tensor(donors[rows], device=states.device)
            shifts = self.strength * (means[donor_rows] - means[source_rows])
            output = states.clone()
            output[source_rows] = torch.where(valid[source_rows, :, None],
                                              states[source_rows] + shifts[:, None, :],
                                              states[source_rows])
            norms[rows] = shifts.norm(dim=1).detach().cpu().numpy()
        for key in np.unique(keys):
            members = keys == key
            record = self.by_pattern[str(int(key))]
            record["blocks"] += int(members.sum())
            record["eligible"] += int((members & eligible).sum())
            record["requested"] += int((members & requested).sum())
            record["selected"] += int((members & selected).sum())
            record["applied"] += int((members & applied).sum())
            record["first_batch_skipped"] += int(members.sum()) if first_batch else 0
            record["shift_l2_sum"] += float(norms[members].sum())
            record["shift_l2_max"] = max(record["shift_l2_max"], float(norms[members].max()))
        # The digest covers every source pattern/utterance and chosen donor, even
        # in the zero-strength control; strength does not affect this record.
        donor_utterances = np.where(eligible, utterances[np.maximum(donors, 0)], -1)
        trace = np.stack((keys, utterances, donors, donor_utterances,
                          requested.astype(np.int64)), axis=1).astype("<i8")
        self._selection_hash.update(np.asarray([optimizer_step, len(keys)], dtype="<i8").tobytes())
        self._selection_hash.update(trace.tobytes())
        self.last_selection = dict(
            optimizer_step=optimizer_step, first_batch_disabled=first_batch,
            pattern_keys=keys.tolist(), utterances=utterances.tolist(),
            donor_rows=donors.tolist(), donor_utterances=donor_utterances.tolist(),
            requested=requested.tolist(), applied=applied.tolist(), shift_l2=norms.tolist())
        return output

    def audit(self):
        counts = ("blocks", "eligible", "requested", "selected", "applied", "first_batch_skipped")
        totals = {name: sum(record[name] for record in self.by_pattern.values()) for name in counts}
        norm_sum = sum(record["shift_l2_sum"] for record in self.by_pattern.values())
        totals.update(shift_l2_sum=norm_sum,
                      shift_l2_mean=norm_sum / max(1, totals["applied"]),
                      shift_l2_max=max((record["shift_l2_max"] for record in self.by_pattern.values()),
                                       default=0.0))
        return dict(batches=self.batches, strength=self.strength, probability=self.probability,
                    selection_sha256=self._selection_hash.hexdigest(), totals=totals,
                    pattern_key_encoding="256 * valid_length + sum(label[t] * 2**t)",
                    by_pattern={key: dict(value) for key, value in sorted(self.by_pattern.items())})
