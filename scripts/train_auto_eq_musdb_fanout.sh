#!/usr/bin/env bash
# Production fan-out for auto-EQ MUSDB per-class models.
#
# Recipe (v18 stack — export-compatible):
#   freeze_freqs=true (C++ split-export precomputes biquad coefficients off
#   fixed band freqs), processor + dyn block_size=128 (matches kBlockSize in
#   native/clap/src/tone_plugin.cpp), lr=3e-3 with cosine warmup (200 steps),
#   grad clip norm=1.0, precision=fp32 (bf16 ComplexHalf NaN'd FFT),
#   2000 max steps, batch_size=64.
#
# Outputs land under /shared/artifacts/auto_eq_musdb_<class>/outputs/<date>/<time>/.

set -e

CLASSES=(full_mix bass drums other vocals)
LOG_DIR=/tmp/auto_eq_fanout_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "Logs -> $LOG_DIR"

for cls in "${CLASSES[@]}"; do
  echo "=== [$(date +%H:%M:%S)] Training class: $cls ==="
  uv run nablafx \
    data=auto_eq_musdb_${cls}_trainval \
    data.batch_size=64 \
    model=gb/tone_auto_eq/model_gb_tone_auto_eq_peq_musdb.d \
    trainer=gb \
    compile=false \
    trainer.precision=32-true \
    trainer.max_steps=2000 \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    model.model.processors.0.freeze_freqs=true \
    model.model.processors.0.block_size=128 \
    model.model.dyn_block_size=128 \
    model.model.dyn_cond_block_size=128 \
    model.lr=3e-3 \
    +model.lr_schedule=cosine_warmup \
    +model.lr_warmup_steps=200 \
    +model.lr_min=1e-4 \
    > "$LOG_DIR/${cls}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] Done: $cls ==="
done

echo "All classes done. Logs in $LOG_DIR"
