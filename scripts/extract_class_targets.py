"""Extract per-class auto-EQ target spectra from MUSDB18-HQ.

For each instrument class (mixture, bass, drums, other, vocals) we compute the
long-term log-mel power spectrum of every track's stem, normalize each track
to zero-mean (so we measure spectral *shape*, not loudness), then take the
median across tracks. The result is a single [n_mels] vector per class — a
"canonical" target shape that the auto-EQ solver pushes input clips toward.

Median (rather than mean) is used so a few outlier tracks (very dark / very
bright mixes) don't skew the reference.

Output:
    weights/auto_eq_refs/<class>.npz with keys:
        class_name           : str
        sample_rate          : int
        n_fft, n_mels        : int
        mel_hz               : [n_mels] band centers in Hz
        target_log_power_db  : [n_mels] zero-mean median log-power, dB
        n_tracks             : int — number of tracks aggregated
    weights/auto_eq_refs/<class>.png — quick visual sanity check.

Usage:
    uv run python scripts/extract_class_targets.py \\
        --src /home/murr/jupyter-redux/datasets/musdb18 \\
        --out weights/auto_eq_refs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from nablafx.data.transforms import _mel_band_log_power


CLASS_TO_STEM = {
    "full_mix": "mixture.wav",
    "bass": "bass.wav",
    "drums": "drums.wav",
    "other": "other.wav",
    "vocals": "vocals.wav",
}


def _load_mono(path: Path, sr_target: int) -> np.ndarray | None:
    try:
        x, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if sr != sr_target:
        t = torch.from_numpy(x).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, sr_target)
        x = t.squeeze(0).numpy().astype(np.float32)
    return x


def _track_log_mel(x: np.ndarray, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Long-term log-mel power [n_mels] for one mono track. Skips silent
    sections by relying on `_mel_band_log_power` averaging over all frames."""
    t = torch.from_numpy(x).view(1, -1)
    _, log_power = _mel_band_log_power(t, sample_rate=sr, n_fft=n_fft, n_mels=n_mels)
    return log_power.squeeze(0).numpy().astype(np.float32)


def _maybe_plot(out_png: Path, mel_hz: np.ndarray, curve_db: np.ndarray,
                per_track: np.ndarray, class_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in per_track:
        ax.plot(mel_hz, row, color="0.7", linewidth=0.4, alpha=0.5)
    ax.plot(mel_hz, curve_db, color="C0", linewidth=2.0, label="median")
    ax.set_xscale("log")
    ax.set_xlabel("Hz (mel band center)")
    ax.set_ylabel("zero-mean log power (dB)")
    ax.set_title(f"auto-EQ target — {class_name} (n={per_track.shape[0]})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MUSDB18 root with train/<song>/")
    ap.add_argument("--out", default="weights/auto_eq_refs",
                    help="Directory to write <class>.npz / <class>.png")
    ap.add_argument("--classes", nargs="+", default=list(CLASS_TO_STEM.keys()),
                    choices=list(CLASS_TO_STEM.keys()),
                    help="Which classes to extract (default: all 5).")
    ap.add_argument("--use-test", action="store_true",
                    help="Include test/ songs in the aggregation (default: train/ only)")
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--n-mels", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_songs = sorted(p for p in (src / "train").iterdir() if p.is_dir()) \
        if (src / "train").is_dir() else []
    test_songs = sorted(p for p in (src / "test").iterdir() if p.is_dir()) \
        if (src / "test").is_dir() else []
    songs = train_songs + (test_songs if args.use_test else [])
    if not songs:
        raise SystemExit(f"no song directories under {src}/train (and /test if --use-test)")
    print(f"aggregating across {len(songs)} song(s) "
          f"({len(train_songs)} train + {len(test_songs) if args.use_test else 0} test)")

    for cls in args.classes:
        stem_name = CLASS_TO_STEM[cls]
        per_track: list[np.ndarray] = []
        for i, song in enumerate(songs):
            stem = song / stem_name
            if not stem.exists():
                print(f"  [{cls}] skip {song.name}: missing {stem_name}")
                continue
            x = _load_mono(stem, args.sample_rate)
            if x is None or x.size < args.n_fft:
                print(f"  [{cls}] skip {song.name}: load failed / too short")
                continue
            log_power = _track_log_mel(x, args.sample_rate, args.n_fft, args.n_mels)
            # Zero-mean: keep shape, drop overall level.
            log_power = log_power - log_power.mean()
            per_track.append(log_power)
            if (i + 1) % 25 == 0:
                print(f"  [{cls}] {i+1}/{len(songs)} processed")

        if not per_track:
            print(f"[{cls}] no tracks aggregated — skipping write")
            continue

        stack = np.stack(per_track, axis=0)  # [n_tracks, n_mels]
        target = np.median(stack, axis=0).astype(np.float32)  # [n_mels]

        # Recover mel_hz the same way _mel_band_log_power does, by calling it on
        # a tiny dummy clip. Cheap and guarantees consistency.
        dummy = torch.zeros(1, max(args.n_fft, args.sample_rate // 4))
        mel_hz, _ = _mel_band_log_power(
            dummy, sample_rate=args.sample_rate,
            n_fft=args.n_fft, n_mels=args.n_mels,
        )
        mel_hz_np = mel_hz.numpy().astype(np.float32)

        out_npz = out / f"{cls}.npz"
        np.savez(
            out_npz,
            class_name=cls,
            sample_rate=np.int64(args.sample_rate),
            n_fft=np.int64(args.n_fft),
            n_mels=np.int64(args.n_mels),
            mel_hz=mel_hz_np,
            target_log_power_db=target,
            n_tracks=np.int64(stack.shape[0]),
        )
        print(f"[{cls}] wrote {out_npz}  (n_tracks={stack.shape[0]}, "
              f"range={target.min():+.1f}..{target.max():+.1f} dB)")

        _maybe_plot(out / f"{cls}.png", mel_hz_np, target, stack, cls)


if __name__ == "__main__":
    main()
