#!/usr/bin/env python3
"""Regenerate native/clap/bench/fixtures/bench_input_20s.wav from MUSDB18-HQ
or stock MUSDB18.

The fixture is committed to git so the bench is hermetic; this script only
exists to document the recipe and to make fixture changes reproducible.
Run this when you want to swap in a different track or duration.

Usage:
  ./prepare_fixture.py [--musdb /path/to/musdb18] [--out fixtures/bench_input_20s.wav]

Defaults are tuned for the dev box layout:
  --musdb ~/jupyter-redux/datasets/musdb18
  --out   <repo>/native/clap/bench/fixtures/bench_input_20s.wav
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


# Track + offset chosen for a stable, mastering-relevant 20 s segment:
# wide spectral content, no silence at start/end, no fades.
TRACK_REL  = "test/Al James - Schoolboy Facination/mixture.wav"
START_SEC  = 30.0
LENGTH_SEC = 20.0
TARGET_SR  = 44100
TARGET_PEAK_DB = -1.0   # peak-normalize so input level is stable run-to-run


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--musdb",
                   default=os.path.expanduser("~/jupyter-redux/datasets/musdb18"),
                   help="path to MUSDB18 root (containing test/, train/)")
    p.add_argument("--out",
                   default=str(here / "fixtures" / "bench_input_20s.wav"),
                   help="output WAV path")
    p.add_argument("--track-rel", default=TRACK_REL,
                   help=f"track path relative to --musdb (default: {TRACK_REL})")
    p.add_argument("--start-sec",  type=float, default=START_SEC)
    p.add_argument("--length-sec", type=float, default=LENGTH_SEC)
    args = p.parse_args()

    src = Path(args.musdb) / args.track_rel
    if not src.is_file():
        print(f"error: {src} not found")
        print(f"       point --musdb at a MUSDB18 (or HQ) checkout")
        return 1

    sr = TARGET_SR
    start = int(args.start_sec  * sr)
    length = int(args.length_sec * sr)

    y, file_sr = sf.read(str(src), start=start, frames=length,
                         always_2d=True, dtype="float32")
    if file_sr != sr:
        print(f"error: expected sr={sr}, got {file_sr} (use MUSDB18-HQ at 44.1k)")
        return 1
    if y.shape[0] != length:
        print(f"error: track is too short for start={args.start_sec}s + length={args.length_sec}s")
        return 1

    peak = float(np.max(np.abs(y)))
    if peak <= 1e-12:
        print(f"error: extracted segment is silent")
        return 1
    y *= (10 ** (TARGET_PEAK_DB / 20.0)) / peak

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, sr, subtype="PCM_16")
    print(f"wrote {out}")
    print(f"  source:   {src}")
    print(f"  start:    {args.start_sec:.3f} s")
    print(f"  duration: {args.length_sec:.3f} s")
    print(f"  sr:       {sr}")
    print(f"  channels: 2")
    print(f"  peak:     {TARGET_PEAK_DB:+.1f} dBFS")
    print(f"  size:     {out.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
