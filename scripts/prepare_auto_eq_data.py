"""Prepare the auto-EQ training dataset for TONE stage 2.

Reads clean broadband audio and writes paired (dry, wet) clips where the wet
version is the result of applying an auto-EQ target transform to the dry.
The network then learns to reproduce that EQ directly from the dry signal.

Output layout matches PluginDataset:
  <out>/DRY/{split}/{name}.input.wav
  <out>/WET/{split}/{name}.target.wav

Usage (synth stand-in corpus while we don't have MUSDB18):
  uv run python scripts/prepare_auto_eq_data.py \\
      --synth --out /shared/datasets/tone_auto_eq \\
      --num-trainval 300 --num-test 30

Usage (real music corpus, generic dir of wavs/flacs — uses brown-noise tilt):
  uv run python scripts/prepare_auto_eq_data.py \\
      --src /path/to/audio_dir \\
      --out /shared/datasets/tone_auto_eq \\
      --clip-seconds 3.0 --num-trainval 400 --num-test 40

Usage (MUSDB18 per-class preset — picks the named stem per song and uses the
matching empirical target spectrum from weights/auto_eq_refs/<class>.npz;
non-overlapping clip-seconds windows):
  uv run python scripts/prepare_auto_eq_data.py --musdb \\
      --src /home/murr/jupyter-redux/datasets/musdb18 \\
      --target-class vocals \\
      --out /shared/datasets/tone_auto_eq_musdb_vocals \\
      --clip-seconds 10.0
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from nablafx.data.transforms import BrownNoiseTargetEQ, EmpiricalTargetEQ
from nablafx.processors.dsp import biquad, sosfilt_via_fsm


_CLASS_TO_STEM = {
    "full_mix": "mixture.wav",
    "bass": "bass.wav",
    "drums": "drums.wav",
    "other": "other.wav",
    "vocals": "vocals.wav",
}


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_24")


def _synth_music_like_clip(rng: np.random.Generator, sr: int, seconds: float) -> np.ndarray:
    """Synthesize a broadband 'music-like' clip: drone + harmonics + noise bed
    + occasional transients. Useful as a stand-in when we don't have MUSDB18
    available for smoke-testing the auto-EQ loop.
    """
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)

    # Chord: random root + 2-4 overtones.
    root = float(rng.uniform(80.0, 300.0))
    n_partials = rng.integers(3, 6)
    for k in range(int(n_partials)):
        f = root * (k + 1) * float(rng.uniform(0.98, 1.02))
        amp = 0.3 / (k + 1) * float(rng.uniform(0.6, 1.2))
        phase = float(rng.uniform(0, 2 * np.pi))
        out += amp * np.sin(2 * np.pi * f * t + phase).astype(np.float32)

    # Noise bed (broadband).
    bed = rng.standard_normal(n).astype(np.float32) * 0.1
    out += bed

    # Occasional transient ticks (drum-like energy across the spectrum).
    for _ in range(int(rng.integers(0, 5))):
        pos = int(rng.uniform(0.1, 0.9) * n)
        length = int(0.02 * sr)
        env = np.exp(-np.arange(length) / (0.006 * sr)).astype(np.float32)
        tick = rng.standard_normal(length).astype(np.float32) * env * 0.5
        end = min(pos + length, n)
        out[pos:end] += tick[: end - pos]

    # Random overall tilt so targets have variety: apply ±6 dB/oct log-freq
    # tilt in the time domain via a one-pole LP or HP emphasis.
    tilt_oct = float(rng.uniform(-3.0, +3.0))
    if abs(tilt_oct) > 0.1:
        alpha = np.clip(np.exp(-abs(tilt_oct) * 0.2), 0.5, 0.99)
        y = np.zeros_like(out)
        acc = 0.0
        for i in range(n):
            acc = alpha * acc + (1.0 - alpha) * out[i]
            y[i] = out[i] - acc if tilt_oct > 0 else acc
        out = y.astype(np.float32)

    # Normalize peak to ~0.5 so the EQ has headroom.
    peak = float(np.max(np.abs(out)) + 1e-8)
    out *= 0.5 / peak
    return out


def _crop_random(x: np.ndarray, want: int, rng: np.random.Generator) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=-1)  # downmix to mono
    if x.shape[0] < want:
        # Pad with zeros to reach the requested length.
        pad = want - x.shape[0]
        return np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    start = rng.integers(0, x.shape[0] - want + 1)
    return x[start : start + want]


def _load_audio(path: str, sr_target: int) -> np.ndarray | None:
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if sr != sr_target:
        # sf.read doesn't resample; cheap nearest-neighbor would be wrong.
        # For now we require input audio to already be at sr_target.
        print(f"  skipping {path}: sr={sr} != {sr_target}")
        return None
    return x


def _load_audio_resampled(path: str, sr_target: int) -> np.ndarray | None:
    """Load any-rate, any-channel audio and return mono float32 at sr_target."""
    try:
        x, sr = torchaudio.load(path, normalize=True, channels_first=True)
    except Exception:
        return None
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != sr_target:
        x = torchaudio.functional.resample(x, sr, sr_target)
    return x.squeeze(0).contiguous().numpy().astype(np.float32)


def _musdb_song_dirs(root: Path) -> list[Path]:
    """Return per-song directories for an unpacked MUSDB18 layout. Accepts
    either `<root>/{train,test}/<song>/...` or `<root>/<song>/...`."""
    dirs: list[Path] = []
    for split in ("train", "test"):
        sd = root / split
        if sd.is_dir():
            dirs.extend(p for p in sorted(sd.iterdir()) if p.is_dir())
    if not dirs:
        dirs.extend(p for p in sorted(root.iterdir()) if p.is_dir())
    return dirs


def _musdb_mixture(song_dir: Path, sr_target: int) -> np.ndarray | None:
    """Load a song's mixture: prefer mixture.wav, else sum the four stems."""
    mix = song_dir / "mixture.wav"
    if mix.exists():
        return _load_audio_resampled(str(mix), sr_target)
    stems = [song_dir / f"{s}.wav" for s in ("bass", "drums", "other", "vocals")]
    if not all(s.exists() for s in stems):
        return None
    parts = [_load_audio_resampled(str(s), sr_target) for s in stems]
    if any(p is None for p in parts):
        return None
    n = min(p.shape[0] for p in parts)
    return np.sum(np.stack([p[:n] for p in parts], axis=0), axis=0).astype(np.float32)


def _musdb_stem(song_dir: Path, target_class: str, sr_target: int) -> np.ndarray | None:
    """Load the stem matching `target_class` from a MUSDB song dir."""
    if target_class == "full_mix":
        return _musdb_mixture(song_dir, sr_target)
    fname = _CLASS_TO_STEM.get(target_class)
    if fname is None:
        return None
    p = song_dir / fname
    if not p.exists():
        return None
    return _load_audio_resampled(str(p), sr_target)


def _augment_pre_eq(
    dry: np.ndarray, sr: int, rng: np.random.Generator,
) -> np.ndarray:
    """Apply a random EQ to the dry signal so the empirical solver sees a
    spectrum that meaningfully differs from the class average.

    Without augmentation, dry stems already mostly sit near the class long-term
    average — the solver computes a near-zero correction on most clips and the
    controller has nothing to discriminate. Pre-EQing dry with random shelves
    and peaks forces real per-clip spectral diversity into the (input, target)
    pairs.

    Composition: random low-shelf + random high-shelf + 1-2 random peaking
    bands. Gains in [-6, +6] dB, Q in [0.5, 1.5] for peaks. Frequencies sampled
    log-uniformly within reasonable bands.
    """
    bands: list[tuple[str, float, float, float]] = []
    bands.append(("low_shelf",  float(rng.uniform(80.0, 250.0)),
                  float(rng.uniform(-6.0, 6.0)), 0.707))
    bands.append(("high_shelf", float(rng.uniform(4000.0, 10000.0)),
                  float(rng.uniform(-6.0, 6.0)), 0.707))
    n_peaks = int(rng.integers(1, 3))
    for _ in range(n_peaks):
        log_f = rng.uniform(np.log10(150.0), np.log10(8000.0))
        bands.append(("peaking", float(10.0 ** log_f),
                      float(rng.uniform(-6.0, 6.0)),
                      float(rng.uniform(0.5, 1.5))))

    x_t = torch.from_numpy(dry).view(1, 1, -1)
    sos_rows: list[torch.Tensor] = []
    for kind, fc, gain_db, q in bands:
        b, a = biquad(
            torch.tensor([gain_db], dtype=torch.float32),
            torch.tensor([fc], dtype=torch.float32),
            torch.tensor([q], dtype=torch.float32),
            sr, kind,
        )
        sos_rows.append(torch.cat((b, a), dim=-1).view(1, 6))
    sos = torch.stack(sos_rows, dim=1)  # [1, n_sos, 6]
    y_t = sosfilt_via_fsm(sos, x_t)
    return y_t.view(-1).numpy().astype(np.float32)


def _segment(x: np.ndarray, win: int, hop: int | None = None) -> list[np.ndarray]:
    """Non-overlapping (hop=win by default) segments of length `win`. Drops
    the trailing partial window."""
    hop = hop or win
    segs: list[np.ndarray] = []
    for start in range(0, x.shape[0] - win + 1, hop):
        segs.append(x[start : start + win])
    return segs


def _run_musdb(
    *,
    src_root: Path,
    out_root: Path,
    sr: int,
    clip_samples: int,
    eq: BrownNoiseTargetEQ,
    target_class: str,
    max_trainval: int | None,
    max_test: int | None,
    rng: np.random.Generator,
    augment_pre_eq: bool = False,
    augment_test: bool = False,
) -> None:
    train_root = src_root / "train"
    test_root = src_root / "test"
    splits: list[tuple[str, list[Path], int | None]] = []
    if train_root.is_dir():
        splits.append(("trainval", _musdb_song_dirs(src_root) if not test_root.is_dir()
                       else [p for p in sorted(train_root.iterdir()) if p.is_dir()],
                       max_trainval))
    if test_root.is_dir():
        splits.append(("test",
                       [p for p in sorted(test_root.iterdir()) if p.is_dir()],
                       max_test))
    if not splits:
        # Flat layout: every subdir is a song; route 90/10 to trainval/test.
        all_songs = _musdb_song_dirs(src_root)
        if not all_songs:
            raise SystemExit(f"no MUSDB-style song directories under {src_root}")
        cut = max(1, int(len(all_songs) * 0.9))
        splits = [("trainval", all_songs[:cut], max_trainval),
                  ("test", all_songs[cut:], max_test)]

    for split, song_dirs, cap in splits:
        dry_dir = out_root / "DRY" / split
        wet_dir = out_root / "WET" / split
        n_written = 0
        n_clipped = 0  # capped songs / windows
        order = list(range(len(song_dirs)))
        rng.shuffle(order)
        for idx_in_split, si in enumerate(order):
            if cap is not None and n_written >= cap:
                n_clipped = len(song_dirs) - idx_in_split
                break
            song = song_dirs[si]
            stem = _musdb_stem(song, target_class, sr)
            if stem is None:
                print(f"  skip {song.name}: no {target_class} stem")
                continue
            segs = _segment(stem, clip_samples)
            n_silent_in_song = 0
            for seg_i, seg in enumerate(segs):
                if cap is not None and n_written >= cap:
                    break
                # Skip near-silent segments — for stems like vocals these are
                # ~12% of the corpus (singer not singing) and they NaN the
                # spectral loss because dry & wet are both zero.
                seg_peak = float(np.max(np.abs(seg)))
                if seg_peak < 1e-4:
                    n_silent_in_song += 1
                    continue
                # Per-segment normalize peak to ~0.5 so EQ has headroom (matches
                # the synth path).
                dry = (seg * (0.5 / seg_peak)).astype(np.float32)

                # Optional pre-EQ augmentation to inflate intra-class spectral
                # variance. Only applied to trainval; test stays clean so the
                # eval reflects natural-signal performance.
                if augment_pre_eq and (split == "trainval" or augment_test):
                    dry = _augment_pre_eq(dry, sr, rng)
                    # Re-normalize so per-segment peak stays ~0.5 after the
                    # augmenting filter (which can boost or cut peak).
                    aug_peak = float(np.max(np.abs(dry)))
                    if aug_peak > 1e-6:
                        dry = (dry * (0.5 / aug_peak)).astype(np.float32)

                x_t = torch.from_numpy(dry).view(1, 1, -1)
                wet_t, gains_db = eq(x_t)
                wet = wet_t.view(-1).numpy().astype(np.float32)

                # If the EQ pushed wet past full-scale, scale dry+wet together
                # to keep wet ≤ 0.95. Hard-clipped wet targets NaN the spectral
                # losses; preserving the relative dry/wet relationship keeps
                # the EQ relationship intact.
                wet_peak = float(np.max(np.abs(wet)))
                if wet_peak > 0.95:
                    scale = 0.95 / wet_peak
                    dry = (dry * scale).astype(np.float32)
                    wet = (wet * scale).astype(np.float32)

                name = f"musdb_{target_class}_{split}_{si:04d}_{seg_i:03d}"
                _write(dry_dir / f"{name}.input.wav", dry, sr)
                _write(wet_dir / f"{name}.target.wav", wet, sr)
                n_written += 1
                if n_written % 100 == 0:
                    print(f"  {split}: {n_written} pairs  "
                          f"(latest: {song.name} seg {seg_i})  "
                          f"gains_db={gains_db[0].tolist()}")

        msg = f"{split}: wrote {n_written} pairs to {out_root}"
        if cap is not None:
            msg += f" (capped at {cap}; {n_clipped} song(s) unused)"
        print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root")
    ap.add_argument("--synth", action="store_true",
                    help="Generate synthetic broadband clips instead of reading --src")
    ap.add_argument("--src", default=None, help="Source directory of .wav files (recurses)")
    ap.add_argument("--musdb", action="store_true",
                    help="Treat --src as an unpacked MUSDB18 root (train/<song>/<stem>.wav). "
                         "Splits MUSDB train→trainval and test→test, resamples, and emits "
                         "non-overlapping clip-seconds windows. Ignores --num-trainval/--num-test "
                         "(uses every available window unless --max-trainval/--max-test set).")
    ap.add_argument("--target-class", choices=list(_CLASS_TO_STEM.keys()), default="full_mix",
                    help="MUSDB only: which stem to read per song and which empirical "
                         "target reference to apply. Loads weights/auto_eq_refs/<class>.npz "
                         "for the wet-side EQ. Default: full_mix.")
    ap.add_argument("--reference-dir", default="weights/auto_eq_refs",
                    help="MUSDB only: directory containing <target-class>.npz references.")
    ap.add_argument("--use-brown", action="store_true",
                    help="MUSDB only: ignore --target-class and use BrownNoiseTargetEQ "
                         "(legacy auto-master target). The stem picked is still --target-class.")
    ap.add_argument("--max-trainval", type=int, default=None,
                    help="MUSDB only: cap on trainval window count.")
    ap.add_argument("--max-test", type=int, default=None,
                    help="MUSDB only: cap on test window count.")
    ap.add_argument("--augment-pre-eq", action="store_true",
                    help="Apply random pre-EQ to dry side of trainval pairs to "
                         "inflate intra-class spectral variance.")
    ap.add_argument("--augment-test", action="store_true",
                    help="Also augment test pairs (default: test stays clean).")
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--clip-seconds", type=float, default=3.0)
    ap.add_argument("--num-trainval", type=int, default=300)
    ap.add_argument("--num-test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    rng = np.random.default_rng(args.seed)
    sr = args.sample_rate
    clip_samples = int(args.clip_seconds * sr)

    if args.musdb:
        if args.src is None:
            raise SystemExit("--src required with --musdb")
        if args.use_brown:
            eq: BrownNoiseTargetEQ = BrownNoiseTargetEQ(sample_rate=sr)
        else:
            ref_path = Path(args.reference_dir) / f"{args.target_class}.npz"
            if not ref_path.exists():
                raise SystemExit(
                    f"reference not found: {ref_path}. Run "
                    f"scripts/extract_class_targets.py first, or pass --use-brown.")
            eq = EmpiricalTargetEQ(str(ref_path), sample_rate=sr)
            print(f"using empirical target: {ref_path} (class={args.target_class})")
        _run_musdb(
            src_root=Path(args.src),
            out_root=out_root,
            sr=sr,
            clip_samples=clip_samples,
            eq=eq,
            target_class=args.target_class,
            max_trainval=args.max_trainval,
            max_test=args.max_test,
            rng=rng,
            augment_pre_eq=args.augment_pre_eq,
            augment_test=args.augment_test,
        )
        return

    # Non-MUSDB paths use the brown-noise target.
    eq = BrownNoiseTargetEQ(sample_rate=sr)

    # If reading real audio, gather file list.
    src_files: list[str] = []
    if not args.synth:
        if args.src is None:
            raise SystemExit("--src required when --synth is not set")
        for ext in ("*.wav", "*.flac"):
            src_files.extend(glob.glob(os.path.join(args.src, "**", ext), recursive=True))
        if not src_files:
            raise SystemExit(f"no audio files found under {args.src}")
        print(f"found {len(src_files)} source files")

    for split, count in [("trainval", args.num_trainval), ("test", args.num_test)]:
        dry_dir = out_root / "DRY" / split
        wet_dir = out_root / "WET" / split
        for i in range(count):
            if args.synth:
                dry = _synth_music_like_clip(rng, sr, args.clip_seconds)
            else:
                path = src_files[int(rng.integers(0, len(src_files)))]
                loaded = _load_audio(path, sr)
                if loaded is None:
                    # try another
                    continue
                dry = _crop_random(loaded, clip_samples, rng).astype(np.float32)

            # Compute wet (brown-noise-tilted).
            x_t = torch.from_numpy(dry).view(1, 1, -1)
            wet_t, gains_db = eq(x_t)
            wet = wet_t.view(-1).numpy().astype(np.float32)

            name = f"ae_{i:05d}"
            _write(dry_dir / f"{name}.input.wav", dry, sr)
            _write(wet_dir / f"{name}.target.wav", wet, sr)

            if (i + 1) % 50 == 0:
                print(f"  {split}: {i+1}/{count}  gains_db={gains_db[0].tolist()}")

        print(f"{split}: wrote {count} pairs to {out_root}")


if __name__ == "__main__":
    main()
