#!/usr/bin/env bash
# Production fan-out for the spectral auto-EQ MUSDB per-class models.
#
# Same training recipe as train_auto_eq_musdb_fanout.sh but uses the
# SpectralDynamicController (Hann-windowed manual rfft -> Linear -> LSTM(64,
# layers=2) -> Linear -> Sigmoid) and the augmented per-class datasets
# (random pre-EQ on the dry side to inflate intra-class spectral variance).
#
# Outputs land under /shared/artifacts/auto_eq_musdb_<class>_aug/outputs/<date>/<time>/.

set -e

CLASSES=(${CLASSES_OVERRIDE:-full_mix bass drums other vocals})
LOG_DIR=/tmp/auto_eq_spectral_fanout_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "Logs -> $LOG_DIR"

for cls in "${CLASSES[@]}"; do
  echo "=== [$(date +%H:%M:%S)] Training class: $cls ==="
  uv run nablafx \
    data=auto_eq_musdb_${cls}_aug_trainval \
    data.batch_size=32 \
    model=gb/tone_auto_eq/model_gb_tone_auto_eq_spectral_musdb.d \
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
