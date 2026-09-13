# Manifests

A manifest is `utt_id \t n_segments \t relpath`, tab-separated, one utterance per
line. `n_segments` is the number of 20 ms segments the scorer expects, and it is
authoritative: `spanmark.predict` takes the segment count from this column and
never derives it from the audio, so a resampled or re-encoded copy cannot
silently change the grid.

`relpath` is relative to `SPANMARK_CORPUS_ROOT`.

## What is here

    ps_train.tsv    25,380 PartialSpoof train utterances
    ps_dev.tsv      24,844 PartialSpoof dev utterances

Both carry the corpus's own identifiers (`CON_T_0020338`), so they resolve
against a stock PartialSpoof download.

## What is not here, and why

The evaluation manifests are **not** shipped. The ones this work was developed
against were anonymised — identifiers were salted to `ps_ad796ad34386280c` and
the audio copied to matching filenames — because the harness treated the
evaluation set as hidden. Those identifiers map to nothing in a stock corpus, so
publishing them would be publishing a file nobody can use.

Build your own instead:

    python scripts/prepare_data.py --corpus-root /path/to/PartialSpoof \
                                   --split eval --labels \
                                   --out assets/manifests/ps_eval.tsv

`--labels` also packs the segment labels into `$SPANMARK_TASKDATA/eval_labels/`,
which `spanmark.evaluate` needs; without it you get a manifest and nothing to
score against.

This walks the corpus, counts segments from each file's duration on the 20 ms
grid, and writes a manifest with the corpus's real identifiers. Numbers produced
against it are directly comparable to the reported ones only if the underlying
audio is the same release — see the Results section of the top-level README.
