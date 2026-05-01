# NeuralMastering

Adaptive mastering chain packaged as a CLAP plugin for macOS. Combines
differentiable DSP blocks with learned controllers (LSTM/RNN) to deliver
per-input adaptive equalization, saturation, dynamics, and limiting.

Built on top of [nablafx](https://github.com/stevemurr/nablafx) (our fork of
[mcomunita/nablafx](https://github.com/mcomunita/nablafx)).

## Stages

In default chain order (re-orderable per-instance via the GUI):

1. **InputLeveler** — LUFS-based input gain matching toward a target.
2. **AutoEQ** — per-class adaptive EQ. Two effector kinds dispatched by
   per-class bundle metadata:
   - `parametric_eq_5band` — 5-band biquad EQ (low-shelf + 3 peaking + high-shelf).
   - `spectral_mask_eq` — STFT-domain magnitude mask (32 or 64 mel-spaced
     bands, minimum-phase reconstruction, frequency + time smoothing).
   Driven by an LSTM controller (`SpectralDynamicController`) that produces
   per-block band gains from a windowed log-magnitude spectrum feature.
   Five class presets: bass / drums / vocals / other / full_mix.
3. **Saturator** — Rational-activation soft-clipper with HPF / threshold /
   bias controls. Pre-/post-gain, wet mix.
4. **Compressor** — LSTM-emulated LA-2A peak reduction.
5. **OutputLeveler** — LUFS leveling at the chain output.
6. **SpatialD** — subtle stereo widening via dimension-D modulation.

A final TruePeakCeiling limiter runs after the user-orderable stages to
guarantee the output never exceeds the configured dBTP ceiling.

## Repo layout

```
neural_mastering/
  export/
    composite.py          ← compose nablafx-export bundles into NM staging
native/clap/              ← C++ CLAP plugin (macOS arm64)
  src/                    ← plugin runtime, DSP blocks, ORT mini-session
  ui/                     ← WebKit GUI
conf/
  data/auto_eq_musdb_*    ← Hydra data configs for training the auto-EQ models
  model/gb/tone_auto_eq/  ← Hydra model configs (parametric + spectral variants)
scripts/
  prepare_auto_eq_data.py    ← MUSDB → (dry, target) pairs with optional augmentation
  extract_class_targets.py   ← per-class long-term spectrum reference extractor
  train_auto_eq_musdb_*.sh   ← per-class training fanout launchers
  probe_auto_eq_adaptivity.py ← post-training adaptivity verification
  check_auto_eq_target_variance.py ← pre-training data-side variance check
  export_tone.py             ← compose trained checkpoints into a staging dir
  build_tone_mac.sh          ← Mac one-shot: stage + build .clap
  install_tone_mac.sh        ← Mac one-shot: build + install to ~/Library/Audio/Plug-Ins/CLAP/
weights/
  auto_eq_refs/           ← per-class long-term spectrum references (.npz)
  tone_bundle/            ← shipped per-stage bundles (model.onnx + plugin_meta.json)
docs/
  mode-collapse.md        ← debugging notes from the auto-EQ controller redesign
```

## Quick start (Mac, end user)

```sh
git clone https://github.com/stevemurr/neural-mastering
cd neural-mastering
bash scripts/install_tone_mac.sh
# Restart your DAW; load "NeuralMastering" from the CLAP plugin list.
```

## Quick start (training new models)

```sh
# 1. Prep an augmented per-class dataset (e.g. full_mix):
uv run python scripts/prepare_auto_eq_data.py --musdb \
    --src /path/to/musdb18 --target-class full_mix --augment-pre-eq \
    --max-trainval 800 --max-test 80 \
    --out /path/to/datasets/tone_auto_eq_musdb_full_mix_aug

# 2. Train (uses the spectral mask 64-band model config by default):
uv run nablafx \
    data=auto_eq_musdb_full_mix_aug_trainval \
    model=gb/tone_auto_eq/model_gb_tone_auto_eq_spectral_mask_2048_musdb.d \
    trainer=gb max_steps=2000

# 3. Probe adaptivity:
uv run python scripts/probe_auto_eq_adaptivity.py --run-dir <hydra_run>

# 4. Export bundle into weights/tone_bundle/auto_eq_<class>/:
uv run nablafx-export --run-dir <hydra_run> --out weights/tone_bundle/auto_eq_full_mix
```

## Architecture notes

- **DAW product name:** "NeuralMastering"
- **Bundle file format:** model.onnx + plugin_meta.json + source.hydra.yaml
  per stage, plus tone_meta.json composing the chain
- **C++ runtime:** ONNX Runtime via a thin `OrtMiniSession` wrapper; LUFS
  leveler / true-peak ceiling / spectral mask EQ are native DSP. Apple
  Accelerate vDSP for FFTs.
- **Why a fork of nablafx:** we extend `nablafx.processors` and
  `nablafx.controllers` with `SpectralMaskEQ` and `SpectralDynamicController`,
  and add the `dynamic-spectral` control_type. Until those land upstream the
  fork is required.

## Status

Active development. The AutoEQ pipeline (parametric and spectral) is
shipping; LA-2A model is shipping; saturator is shipping; compressor work
is in progress.
