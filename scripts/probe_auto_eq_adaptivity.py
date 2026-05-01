"""Probe a trained auto-EQ controller for adaptivity.

Loads the trained controller checkpoint, runs it over a battery of
spectrally-distinct synthetic test signals, and reports the predicted EQ
gain vector per signal. If the model is collapsed, all signals produce ~the
same gains. If adaptive, the gains differ meaningfully.

Usage:
  uv run python scripts/probe_auto_eq_adaptivity.py \
      --run-dir /shared/artifacts/auto_eq_musdb_bass_aug/outputs/<date>/<time>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nablafx.export.bundle import _load_system_and_weights  # type: ignore[attr-defined]


_BAND_NAMES = ["LS@1010", "P@110", "P@1100", "P@7000", "HS@10000"]
_GAIN_CHANNELS = [0, 3, 6, 9, 12]   # gain channel indices in the 15-param vector
_GAIN_MIN, _GAIN_MAX = -9.0, 9.0


def _make_test_signals(sr: int = 44100, dur: float = 3.0) -> dict[str, np.ndarray]:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(0)

    sigs: dict[str, np.ndarray] = {}

    def _peak_norm(x: np.ndarray, level: float = 0.5) -> np.ndarray:
        p = float(np.max(np.abs(x)) + 1e-9)
        return (x * (level / p)).astype(np.float32)

    sigs["sub_bass_60Hz"] = _peak_norm(np.sin(2 * np.pi * 60.0 * t).astype(np.float32))
    sigs["bass_120Hz"]    = _peak_norm(np.sin(2 * np.pi * 120.0 * t).astype(np.float32))
    sigs["mid_1kHz"]      = _peak_norm(np.sin(2 * np.pi * 1000.0 * t).astype(np.float32))
    sigs["high_8kHz"]     = _peak_norm(np.sin(2 * np.pi * 8000.0 * t).astype(np.float32))
    sigs["white_noise"]   = _peak_norm(rng.standard_normal(n).astype(np.float32))

    # 1/f pink-ish noise via cumulative sum
    pink = np.cumsum(rng.standard_normal(n).astype(np.float32))
    pink -= pink.mean()
    sigs["pink_noise"] = _peak_norm(pink)

    # Brown-ish (heavier low end)
    brown = np.cumsum(np.cumsum(rng.standard_normal(n).astype(np.float32)))
    brown -= brown.mean()
    sigs["brown_noise"] = _peak_norm(brown)

    # Filtered noise with bright tilt
    bright = rng.standard_normal(n).astype(np.float32)
    # crude high-pass via difference
    bright = np.diff(bright, prepend=0.0)
    sigs["bright_hpf_noise"] = _peak_norm(bright)

    return sigs


def _run_controller_on_clip(ctrl: torch.nn.Module, x: np.ndarray, block_size: int = 128) -> np.ndarray:
    """Run the controller block-by-block (as the C++ runtime would) and return
    the per-block 15-channel gain vector."""
    ctrl.eval()
    if hasattr(ctrl, "reset_state"):
        ctrl.reset_state()
    n = x.shape[0]
    n = (n // block_size) * block_size
    x = x[:n]
    n_blocks = n // block_size
    gains_per_block = np.zeros((n_blocks, 5), dtype=np.float32)
    with torch.no_grad():
        for b in range(n_blocks):
            blk = x[b * block_size : (b + 1) * block_size]
            blk_t = torch.from_numpy(blk).view(1, 1, block_size)
            params = ctrl(blk_t)  # [1, 15, block_size]
            params0 = params[0, :, 0].numpy()  # one block-step
            for k, ch in enumerate(_GAIN_CHANNELS):
                gains_per_block[b, k] = _GAIN_MIN + params0[ch] * (_GAIN_MAX - _GAIN_MIN)
    return gains_per_block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Hydra run output directory")
    ap.add_argument("--ckpt", default=None, help="Optional explicit ckpt path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    system = _load_system_and_weights(run_dir, ckpt_path=Path(args.ckpt) if args.ckpt else None)
    model = system.model
    ctrl = model.controller.controllers[0]
    print(f"loaded controller: {type(ctrl).__name__}")
    print(f"  block_size={getattr(ctrl, 'block_size', '?')}, "
          f"hidden={getattr(getattr(ctrl, 'lstm', None), 'hidden_size', '?')}, "
          f"layers={getattr(getattr(ctrl, 'lstm', None), 'num_layers', '?')}")

    sigs = _make_test_signals()
    rows: dict[str, np.ndarray] = {}
    print()
    print(f"Per-signal mean gain_db (averaged over all blocks):")
    print(f"{'signal':<22}" + " ".join(f"{n:>9}" for n in _BAND_NAMES))
    for name, x in sigs.items():
        gpb = _run_controller_on_clip(ctrl, x)
        mean = gpb.mean(axis=0)
        rows[name] = mean
        print(f"{name:<22}" + " ".join(f"{g:>9.2f}" for g in mean))

    M = np.stack(list(rows.values()))  # [n_sigs, 5]
    cross_std = M.std(axis=0)
    print()
    print(f"Across-signal std per band (higher = more adaptive):")
    print("                      " + " ".join(f"{s:>9.2f}" for s in cross_std))
    overall = float(np.linalg.norm(cross_std))
    print()
    print(f"||per-band cross-signal std||_2 = {overall:.2f} dB")
    if overall < 1.0:
        verdict = "COLLAPSED — outputs nearly identical across signal types"
    elif overall < 3.0:
        verdict = "WEAKLY ADAPTIVE — some response to material"
    else:
        verdict = "ADAPTIVE — model meaningfully differentiates spectral content"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
