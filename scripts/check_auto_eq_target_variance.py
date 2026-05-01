"""Probe target diversity for auto-EQ training (mode-collapse Fix 5).

Walks the prepared DRY clips of a class, re-runs the EmpiricalTargetEQ solver
that produced the WET targets, and reports the per-band gain distribution.

If the std of `gains_db` across the corpus is small (<~1 dB per band), the
training targets themselves are nearly constant — the model has nothing to
learn beyond the class mean and architectural fixes won't help. If the std is
large, the data has variance and the collapse is downstream (model capacity,
context, or runtime normalization mismatch).

Usage:
  uv run python scripts/check_auto_eq_target_variance.py \
      --class bass \
      --dataset-root /shared/datasets/tone_auto_eq_musdb_bass \
      --reference weights/auto_eq_refs/bass.npz \
      [--max-clips 500]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nablafx.data.transforms import EmpiricalTargetEQ


_BAND_NAMES = ["LS@1010", "P@110", "P@1100", "P@7000", "HS@10000"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True,
                    help="Class name (bass/drums/vocals/other/full_mix) — labelling only")
    ap.add_argument("--dataset-root", required=True,
                    help="e.g. /shared/datasets/tone_auto_eq_musdb_bass")
    ap.add_argument("--reference", required=True,
                    help="e.g. weights/auto_eq_refs/bass.npz")
    ap.add_argument("--split", default="trainval", choices=["trainval", "test"])
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    dry_dir = Path(args.dataset_root) / "DRY" / args.split
    files = sorted(dry_dir.glob("*.input.wav"))
    if not files:
        raise SystemExit(f"no .input.wav under {dry_dir}")
    if args.max_clips is not None:
        files = files[: args.max_clips]
    print(f"class={args.cls}  split={args.split}  n_clips={len(files)}")
    print(f"reference={args.reference}")

    eq = EmpiricalTargetEQ(args.reference, sample_rate=args.sample_rate)

    gains = np.zeros((len(files), 5), dtype=np.float32)
    for i, f in enumerate(files):
        x, sr = sf.read(str(f), dtype="float32")
        if sr != args.sample_rate:
            raise SystemExit(f"{f} sr={sr} != {args.sample_rate}")
        if x.ndim > 1:
            x = x.mean(axis=-1)
        x_t = torch.from_numpy(x).view(1, 1, -1)
        _, g = eq(x_t)
        gains[i] = g.view(-1).numpy()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    mean = gains.mean(axis=0)
    std = gains.std(axis=0)
    p10 = np.percentile(gains, 10, axis=0)
    p50 = np.percentile(gains, 50, axis=0)
    p90 = np.percentile(gains, 90, axis=0)
    rng = gains.max(axis=0) - gains.min(axis=0)

    print()
    print(f"Per-band gain_db distribution across {len(files)} clips")
    print(f"{'band':<10} {'mean':>7} {'std':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'range':>7}")
    for k, name in enumerate(_BAND_NAMES):
        print(f"{name:<10} {mean[k]:>7.2f} {std[k]:>7.2f} "
              f"{p10[k]:>7.2f} {p50[k]:>7.2f} {p90[k]:>7.2f} {rng[k]:>7.2f}")

    overall_std = float(np.linalg.norm(std))
    print()
    print(f"||per-band std||_2 = {overall_std:.2f} dB")
    if overall_std < 2.0:
        verdict = ("LOW variance: targets are nearly constant. The model has little "
                   "to learn beyond the class mean. Add input pre-EQ augmentation "
                   "before chasing architectural fixes.")
    elif overall_std < 5.0:
        verdict = ("MODERATE variance: real diversity exists. If the model still "
                   "collapses, suspect runtime normalization mismatch or insufficient "
                   "model capacity.")
    else:
        verdict = ("HIGH variance: lots of signal to learn. Collapse is almost "
                   "certainly downstream (runtime normalization, model capacity, "
                   "or training optimization).")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
