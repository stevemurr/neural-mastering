# tone_bench

Headless CLAP host for benchmarking the NeuralMastering composite plugin.
Loads a `.clap` bundle, drives a fixed input WAV through it in fixed-size
blocks, and reports per-block timing (p50/p95/p99/max) plus RTF.

## Build

`tone_bench` builds as a sibling target to `tone_clap` whenever libsndfile is
installed. It's optional — the plugin builds fine without it.

```
brew install libsndfile
./build.sh tone <staging_dir> /tmp/NeuralMastering.clap   # builds tone_bench too
```

The binary lands at `native/clap/build/tone_bench`.

## Run

```
./tone_bench \
    --plugin /tmp/NeuralMastering.clap \
    --in     bench-input.wav \
    --out    /tmp/out.wav \
    --buffer 256 \
    --iters  20 \
    --warmup 2 \
    --params 'EQ=1.0,EQR=1.0,SSC=1.0,SDR=12,SVO=0,SMX=0.5,STH=-12,CMP=50,CLS=4,LVL=1' \
    --json
```

JSON output:

```json
{
  "plugin": "NeuralMastering",
  "sample_rate": 44100,
  "buffer_size": 256,
  "channels": 2,
  "iters": 20,
  "warmup": 2,
  "frames_per_iter": 1323000,
  "audio_seconds_per_iter": 30.0,
  "per_block_us": {"min": 410.2, "p50": 482.1, "p95": 612.8, "p99": 851.4, "max": 1203.0, "mean": 503.6, "count": 103580},
  "per_iter_seconds_mean": 4.214,
  "realtime_factor_mean": 7.12,
  "block_deadline_us": 5805.4,
  "deadline_miss_count": 0,
  "deadline_miss_pct": 0.0
}
```

## What the numbers mean

- **per-block us** — wall time of one `plug->process()` call. **p99 is the
  realtime-deadline number.** If p99 exceeds `block_deadline_us`, the audio
  thread will glitch in a DAW at this buffer size.
- **block_deadline_us** — `buffer_size / sr * 1e6`. The wall-clock budget
  for one block.
- **deadline_miss_count / pct** — blocks slower than the deadline. Should be 0.
- **RTF** — `audio_seconds / wall_seconds` averaged over iters. >1 = faster
  than realtime; <1 = unable to keep up offline (much worse for realtime).

## Param IDs

`--params` takes the **short IDs** (3-letter symbols, not the human-readable
names). The bench computes the CLAP integer ID via the same FNV hash the
plugin uses (`param_id_for(effect_name, short_id)`).

Common short IDs (full list in `composite.py::_build_default_meta`):

```
LVL  Leveler           [0..1]
LVT  Lev Target  LUFS  [-36..-6]
SDR  Sat Drive   dB    [0..24]
SVO  Sat Output  dB    [-24..12]
SMX  Sat Mix           [0..1]
SHF  Sat HPF     Hz    [20..500]
STH  Sat Thresh  dB    [-24..0]
SBS  Sat Bias          [-0.5..0.5]
C_L  Comp/Limit        [0..1]
CMP  Peak Reduction    [0..100]
SSC  Bus Comp          [0..1]
CLS  EQ Class    enum  [0..n_classes-1]   (4 = full_mix)
EQ   Auto EQ           [0..1]
EQR  EQ Range          [0..1]
EQS  EQ Speed    ms    [10..500]
EQB  EQ Boost          [0..1]
OLV  Out Leveler       [0..1]
OLT  Out Lev Target    [-36..-6]
TRM  Output Trim dB    [-12..12]
```

## Stage-isolation pattern

Set every stage's amount to 0 except the one under test:

```
# SSL bus comp only
--params 'EQ=0,SDR=0,CMP=0,SSC=1.0,LVL=0,OLV=0'

# Auto-EQ only (full_mix class)
--params 'EQ=1.0,SDR=0,CMP=0,SSC=0,CLS=4,LVL=0,OLV=0'
```

`per_block_us` then attributes the cost to the active stage. Combined with
Instruments' time profiler attached to the running `tone_bench` process,
you get both stage-level timing and per-function attribution.
