# tone_bench — performance test suite

Headless CLAP host + scripted scenario matrix for benchmarking the
NeuralMastering composite plugin. Designed so an agent (or CI job) can
run the whole suite with one command and either dump JSON or diff
against a committed baseline.

## Layout

```
bench/
  tone_bench.cpp      C++ harness: dlopens a .clap, drives audio in fixed
                      blocks, emits per-block timing as JSON
  run_bench.py        Python runner: runs scenarios × buffer sizes, writes
                      last_run.json + last_run.md, optionally diffs vs
                      baseline.json
  prepare_fixture.py  Regenerates fixtures/bench_input_20s.wav from MUSDB18
                      (only needed if you want to change the fixture)
  fixtures/
    bench_input_20s.wav   committed 20 s mastering-grade stereo input
                           (44.1 kHz, peak-normalized to -1 dBFS)
  baseline.json       committed reference numbers; populated by
                      `run_bench.py --update-baseline`
  last_run.json       gitignored; latest results dump
  last_run.md         gitignored; latest human-readable summary
```

## One-shot run (the agent path)

```bash
brew install libsndfile pkg-config         # one-time
./build.sh tone <staging-dir> /tmp/NeuralMastering.clap
./native/clap/bench/run_bench.py
```

Exit codes:
- `0` — pass (no regression vs baseline, or no baseline to compare against)
- `1` — setup error (missing binary / bundle / fixture)
- `2` — regression detected (p99 > baseline + 15% **or** RTF < baseline − 10%
         **or** any new deadline miss)

`last_run.md` is the human-readable summary. `last_run.json` is the
machine-readable dump with every cell's stats.

## Scenario × buffer matrix

| scenario        | what's exercised                                        |
|-----------------|---------------------------------------------------------|
| `full_chain`    | every stage active at moderate-to-high amounts          |
| `eq_only`       | just the auto-EQ (controller + SpectralMaskEq)          |
| `sat_only`      | just the saturator (RationalA)                          |
| `la2a_only`     | just the LA-2A LSTM                                     |
| `ssl_comp_only` | just the SSL bus comp TCN                               |
| `bypass`        | every stage's amount at 0 — measures plumbing overhead  |

Buffers: `64, 128, 256, 512, 1024` frames @ 44.1 kHz (deadlines:
1.45, 2.90, 5.81, 11.61, 23.22 ms).

Default cell config: 10 timed iters + 2 warmup. Override with
`--iters N --warmup N` if you want quicker feedback or tighter percentiles.

## Subset runs

```bash
# Just the SSL comp on small buffers
./run_bench.py --scenarios ssl_comp_only --buffers 64,128

# Faster iteration: 3 iters, no warmup, no compare
./run_bench.py --iters 3 --warmup 0 --no-compare
```

## Updating the baseline

After landing a perf change you believe is a real improvement:

```bash
./native/clap/bench/run_bench.py --update-baseline
git add native/clap/bench/baseline.json
git commit -m "perf: update bench baseline (<reason>)"
```

The baseline is committed so PR review can see the diff; never overwrite
it without explaining why in the commit message.

## Direct `tone_bench` invocation

For ad-hoc profiling (e.g. when attaching Instruments), call the binary
directly:

```bash
native/clap/build/tone_bench \
    --plugin /tmp/NeuralMastering.clap \
    --in     native/clap/bench/fixtures/bench_input_20s.wav \
    --buffer 256 --iters 200 --warmup 5 \
    --params 'EQ=0,SDR=0,CMP=0,SSC=1.0,CLS=4,LVL=0,OLV=0' \
    --json
```

JSON shape:

```json
{
  "plugin": "NeuralMastering",
  "sample_rate": 44100,
  "buffer_size": 256,
  "channels": 2,
  "iters": 200,
  "warmup": 5,
  "frames_per_iter": 882000,
  "audio_seconds_per_iter": 20.0,
  "per_block_us": {"min": 410.2, "p50": 482.1, "p95": 612.8, "p99": 851.4, "max": 1203.0, "mean": 503.6, "count": 690000},
  "per_iter_seconds_mean": 0.347,
  "realtime_factor_mean": 57.6,
  "block_deadline_us": 5805.4,
  "deadline_miss_count": 0,
  "deadline_miss_pct": 0.0
}
```

### What the numbers mean

- **per_block_us.p99** — wall time of the 99th-percentile `process()`
  call. **This is the realtime-deadline number**: if p99 exceeds
  `block_deadline_us`, the audio thread will glitch in a DAW at that
  buffer size.
- **block_deadline_us** — `buffer_size / sr * 1e6`. The wall-clock
  budget for one block.
- **deadline_miss_count / pct** — blocks slower than the deadline.
  Should be `0`.
- **realtime_factor_mean** — `audio_seconds / wall_seconds` averaged
  over iters. >1 = faster than realtime offline. (Realtime DAW
  playback wants RTF ≫ 1 because the audio thread shares the core.)

## Param IDs

`--params` takes the **short IDs** (3-letter symbols, not human names).
The bench computes the CLAP integer param id via the same FNV-1a hash
the plugin uses (`param_id_for(effect_name, short_id)`).

```
LVL  Leveler                  [0..1]
LVT  Lev Target  LUFS         [-36..-6]
SDR  Sat Drive   dB           [0..24]
SVO  Sat Output  dB           [-24..12]
SMX  Sat Mix                  [0..1]
SHF  Sat HPF     Hz           [20..500]
STH  Sat Thresh  dB           [-24..0]
SBS  Sat Bias                 [-0.5..0.5]
C_L  Comp/Limit               [0..1]
CMP  Peak Reduction           [0..100]
SSC  Bus Comp                 [0..1]
CLS  EQ Class    enum         [0..n_classes-1]   (4 = full_mix)
EQ   Auto EQ                  [0..1]
EQR  EQ Range                 [0..1]
EQS  EQ Speed    ms           [10..500]
EQB  EQ Boost                 [0..1]
OLV  Out Leveler              [0..1]
OLT  Out Lev Target  LUFS     [-36..-6]
TRM  Output Trim dB           [-12..12]
```
