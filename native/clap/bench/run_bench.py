#!/usr/bin/env python3
"""Run the NeuralMastering performance suite.

Drives `tone_bench` over a fixed scenario × buffer-size matrix, collects
per-block timing stats, and emits both a JSON dump and a Markdown summary.
Optionally diffs against a committed baseline.json so regressions are
visible in CI / PR review.

Designed to be runnable by an agent with one command:

    ./native/clap/bench/run_bench.py

The defaults are wired for a dev box that has just run `./build.sh tone`:
  --bench-bin   native/clap/build/tone_bench
  --bundle      auto-discover (./build, /tmp, ~/Library/Audio/Plug-Ins/CLAP)
  --fixture     native/clap/bench/fixtures/bench_input_20s.wav
  --baseline    native/clap/bench/baseline.json (compared if it exists)
  --out         native/clap/bench/last_run.json (always overwritten)

Pass --update-baseline to write the current run as the new baseline.
Exit codes: 0 = pass, 1 = setup error, 2 = regression detected.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent  # native/clap/bench/ -> repo root


# ----------------------------------------------------------------------------
# Scenario matrix
# ----------------------------------------------------------------------------
#
# Each scenario sets the host-exposed control values so a single stage (or the
# full chain) is exercised. Buffer sizes span what real hosts use: 64 (Logic
# at low latency) up through 1024 (offline render).

# Wet stages: amounts at moderate-to-high values so we see realistic load.
# CLS=4 selects full_mix, the most-tested auto-EQ class.
FULL_CHAIN = "EQ=1.0,EQR=1.0,EQB=1.0,EQS=100,SDR=12,SVO=0,SMX=1.0,STH=-12,SBS=0,CMP=50,SSC=1.0,CLS=4,LVL=1.0,LVT=-14,OLV=1.0,OLT=-14,TRM=0"

# Single-stage isolations: zero everything else so attribution is clean.
EQ_ONLY      = "EQ=1.0,EQR=1.0,EQB=1.0,EQS=100,SDR=0,SVO=0,SMX=0,STH=0,SBS=0,CMP=0,SSC=0,CLS=4,LVL=0,OLV=0,TRM=0"
SAT_ONLY     = "EQ=0,SDR=12,SVO=0,SMX=1.0,STH=-12,SBS=0,CMP=0,SSC=0,CLS=4,LVL=0,OLV=0,TRM=0"
LA2A_ONLY    = "EQ=0,SDR=0,SVO=0,SMX=0,STH=0,SBS=0,CMP=50,SSC=0,CLS=4,LVL=0,OLV=0,TRM=0"
SSL_COMP_ONLY = "EQ=0,SDR=0,SVO=0,SMX=0,STH=0,SBS=0,CMP=0,SSC=1.0,CLS=4,LVL=0,OLV=0,TRM=0"

# Bypass: every stage's amount at 0 to measure plumbing overhead only.
BYPASS = "EQ=0,SDR=0,SVO=0,SMX=0,STH=0,SBS=0,CMP=0,SSC=0,CLS=4,LVL=0,OLV=0,TRM=0"

SCENARIOS: List[tuple[str, str]] = [
    ("full_chain",    FULL_CHAIN),
    ("eq_only",       EQ_ONLY),
    ("sat_only",      SAT_ONLY),
    ("la2a_only",     LA2A_ONLY),
    ("ssl_comp_only", SSL_COMP_ONLY),
    ("bypass",        BYPASS),
]

BUFFER_SIZES = [64, 128, 256, 512, 1024]
DEFAULT_ITERS = 10
DEFAULT_WARMUP = 2

# Regression thresholds — fail the run if p99 or RTF moves more than this
# fraction in the bad direction relative to baseline.
P99_REGRESS_FRAC = 0.15   # +15% latency
RTF_REGRESS_FRAC = 0.10   # -10% throughput


# ----------------------------------------------------------------------------
# Bundle discovery
# ----------------------------------------------------------------------------

def discover_bundle() -> Optional[Path]:
    candidates = [
        REPO_ROOT / "build" / "NeuralMastering.clap",
        Path("/tmp/NeuralMastering.clap"),
        Path(os.path.expanduser("~/Library/Audio/Plug-Ins/CLAP/NeuralMastering.clap")),
    ]
    for c in candidates:
        if (c / "Contents" / "MacOS").is_dir():
            return c
    return None


def discover_bench_bin() -> Optional[Path]:
    p = REPO_ROOT / "native" / "clap" / "build" / "tone_bench"
    return p if p.is_file() else None


# ----------------------------------------------------------------------------
# Run one tone_bench invocation
# ----------------------------------------------------------------------------

@dataclass
class CellResult:
    scenario: str
    buffer_size: int
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    mean_us: float
    block_count: int
    rtf: float
    deadline_us: float
    deadline_misses: int
    deadline_miss_pct: float


def run_cell(bench_bin: Path, bundle: Path, fixture: Path,
             scenario: str, params: str,
             buffer_size: int, iters: int, warmup: int) -> CellResult:
    cmd = [
        str(bench_bin),
        "--plugin",  str(bundle),
        "--in",      str(fixture),
        "--buffer",  str(buffer_size),
        "--iters",   str(iters),
        "--warmup",  str(warmup),
        "--params",  params,
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[{scenario} buf={buffer_size}] tone_bench failed:\n")
        sys.stderr.write(proc.stderr)
        sys.exit(1)
    j = json.loads(proc.stdout)
    pb = j["per_block_us"]
    return CellResult(
        scenario=scenario, buffer_size=buffer_size,
        p50_us=pb["p50"], p95_us=pb["p95"], p99_us=pb["p99"],
        max_us=pb["max"], mean_us=pb["mean"], block_count=pb["count"],
        rtf=j["realtime_factor_mean"],
        deadline_us=j["block_deadline_us"],
        deadline_misses=j["deadline_miss_count"],
        deadline_miss_pct=j["deadline_miss_pct"],
    )


# ----------------------------------------------------------------------------
# Stats / diff
# ----------------------------------------------------------------------------

def cell_to_dict(c: CellResult) -> dict:
    return {
        "scenario": c.scenario, "buffer_size": c.buffer_size,
        "p50_us": c.p50_us, "p95_us": c.p95_us, "p99_us": c.p99_us,
        "max_us": c.max_us, "mean_us": c.mean_us, "block_count": c.block_count,
        "rtf": c.rtf, "deadline_us": c.deadline_us,
        "deadline_misses": c.deadline_misses,
        "deadline_miss_pct": c.deadline_miss_pct,
    }


def index_results(results: List[CellResult]) -> Dict[tuple[str, int], CellResult]:
    return {(r.scenario, r.buffer_size): r for r in results}


@dataclass
class Diff:
    scenario: str
    buffer_size: int
    p99_delta_pct: float        # positive = slower
    rtf_delta_pct: float        # positive = faster (good)
    new_misses: int             # >0 = new deadline misses
    regression: bool            # exceeds thresholds


def compare(now: List[CellResult], base: List[CellResult]) -> List[Diff]:
    base_idx = index_results(base)
    out: List[Diff] = []
    for r in now:
        b = base_idx.get((r.scenario, r.buffer_size))
        if b is None:
            continue
        p99_delta = (r.p99_us - b.p99_us) / b.p99_us * 100.0 if b.p99_us > 0 else 0.0
        rtf_delta = (r.rtf  - b.rtf)  / b.rtf  * 100.0 if b.rtf  > 0 else 0.0
        new_misses = max(0, r.deadline_misses - b.deadline_misses)
        regression = (
            p99_delta > P99_REGRESS_FRAC * 100.0
            or rtf_delta < -RTF_REGRESS_FRAC * 100.0
            or new_misses > 0
        )
        out.append(Diff(r.scenario, r.buffer_size, p99_delta, rtf_delta, new_misses, regression))
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def format_summary(now: List[CellResult],
                   diffs: Optional[List[Diff]],
                   meta: dict) -> str:
    lines: List[str] = []
    lines.append(f"# tone_bench results — {meta['plugin']}")
    lines.append("")
    lines.append(f"- bundle: `{meta['bundle']}`")
    lines.append(f"- fixture: `{meta['fixture']}` ({meta['fixture_sec']:.1f}s @ {meta['sample_rate']} Hz)")
    lines.append(f"- iters/cell: {meta['iters']}, warmup: {meta['warmup']}")
    lines.append(f"- block-deadline regression threshold: p99 +{int(P99_REGRESS_FRAC*100)}%, RTF -{int(RTF_REGRESS_FRAC*100)}%")
    lines.append("")

    diff_idx: Dict[tuple[str, int], Diff] = {}
    if diffs:
        diff_idx = {(d.scenario, d.buffer_size): d for d in diffs}

    by_scenario: Dict[str, List[CellResult]] = {}
    for r in now:
        by_scenario.setdefault(r.scenario, []).append(r)

    for scenario, rs in by_scenario.items():
        rs.sort(key=lambda r: r.buffer_size)
        lines.append(f"## `{scenario}`")
        lines.append("")
        if diffs is not None:
            lines.append("| buf | p50 µs | p99 µs | max µs | RTF    | misses |  Δp99  |  ΔRTF  | flag |")
            lines.append("|----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|:----:|")
        else:
            lines.append("| buf | p50 µs | p99 µs | max µs | RTF    | misses |")
            lines.append("|----:|-------:|-------:|-------:|-------:|-------:|")
        for r in rs:
            row = f"| {r.buffer_size:>3} | {r.p50_us:6.1f} | {r.p99_us:6.1f} | {r.max_us:6.1f} | {r.rtf:6.2f}× | {r.deadline_misses:>6d}"
            if diffs is not None:
                d = diff_idx.get((r.scenario, r.buffer_size))
                if d is not None:
                    flag = "🔴" if d.regression else ("🆕" if d.new_misses > 0 else "✓")
                    row += f" | {d.p99_delta_pct:+6.1f}% | {d.rtf_delta_pct:+6.1f}% | {flag}"
                else:
                    row += " |    —   |    —   |  🆕  "
            row += " |"
            lines.append(row)
        lines.append("")

    if diffs is not None:
        regressions = [d for d in diffs if d.regression]
        if regressions:
            lines.append(f"## ⚠️  {len(regressions)} regression(s) vs baseline")
            for d in regressions:
                lines.append(
                    f"- `{d.scenario}` buf={d.buffer_size}: p99 {d.p99_delta_pct:+.1f}%, "
                    f"RTF {d.rtf_delta_pct:+.1f}%, new misses={d.new_misses}"
                )
        else:
            lines.append("## ✓ No regressions vs baseline")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-bin", type=Path, default=None,
                   help="path to tone_bench binary")
    p.add_argument("--bundle", type=Path, default=None,
                   help="path to NeuralMastering.clap bundle")
    p.add_argument("--fixture", type=Path,
                   default=HERE / "fixtures" / "bench_input_20s.wav")
    p.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    p.add_argument("--out", type=Path, default=HERE / "last_run.json")
    p.add_argument("--summary", type=Path, default=HERE / "last_run.md")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--buffers", type=str, default=None,
                   help=f"comma-separated buffer sizes (default: {','.join(map(str, BUFFER_SIZES))})")
    p.add_argument("--scenarios", type=str, default=None,
                   help="comma-separated scenarios to run (default: all)")
    p.add_argument("--update-baseline", action="store_true",
                   help="write current run to --baseline (no comparison)")
    p.add_argument("--no-compare", action="store_true",
                   help="do not compare against --baseline even if it exists")
    args = p.parse_args()

    if sys.platform != "darwin":
        print("error: tone_bench is macOS-arm64 only (see CMakeLists.txt)")
        return 1

    bench_bin = args.bench_bin or discover_bench_bin()
    if bench_bin is None or not bench_bin.is_file():
        print(f"error: tone_bench binary not found at {bench_bin or '<auto>'}")
        print(f"       build with: ./build.sh tone <staging> /tmp/NeuralMastering.clap")
        return 1
    bundle = args.bundle or discover_bundle()
    if bundle is None or not (bundle / "Contents" / "MacOS").is_dir():
        print(f"error: NeuralMastering.clap bundle not found at {bundle or '<auto>'}")
        print(f"       searched: ./build, /tmp, ~/Library/Audio/Plug-Ins/CLAP")
        return 1
    if not args.fixture.is_file():
        print(f"error: fixture not found: {args.fixture}")
        print(f"       regenerate with: ./prepare_fixture.py")
        return 1

    buffers = (BUFFER_SIZES if args.buffers is None
               else [int(x) for x in args.buffers.split(",") if x.strip()])
    scenarios = SCENARIOS
    if args.scenarios is not None:
        wanted = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in SCENARIOS if s[0] in wanted]
        if not scenarios:
            print(f"error: no scenarios matched: {args.scenarios}")
            return 1

    # Probe sample rate from the fixture file (so the report stays accurate
    # if someone swaps the fixture for a 48 kHz one).
    import wave
    with wave.open(str(args.fixture), "rb") as wf:
        sample_rate = wf.getframerate()
        fixture_sec = wf.getnframes() / float(sample_rate)

    print(f"bench_bin: {bench_bin}")
    print(f"bundle:    {bundle}")
    print(f"fixture:   {args.fixture}  ({fixture_sec:.1f}s @ {sample_rate} Hz)")
    print(f"matrix:    {len(scenarios)} scenarios × {len(buffers)} buffers"
          f" = {len(scenarios)*len(buffers)} cells, {args.iters} iters each")
    print()

    results: List[CellResult] = []
    for name, params in scenarios:
        for buf in buffers:
            print(f"  [{name:<14s}  buf={buf:>4d}] ", end="", flush=True)
            r = run_cell(bench_bin, bundle, args.fixture, name, params,
                         buf, args.iters, args.warmup)
            results.append(r)
            print(f"p99={r.p99_us:7.1f} µs  RTF={r.rtf:5.2f}×  "
                  f"misses={r.deadline_misses:>3d}/{r.block_count}")
    print()

    meta = {
        "plugin": "NeuralMastering",
        "bundle": str(bundle),
        "fixture": str(args.fixture),
        "fixture_sec": fixture_sec,
        "sample_rate": sample_rate,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    payload = {"meta": meta, "results": [cell_to_dict(r) for r in results]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")

    diffs: Optional[List[Diff]] = None
    if args.update_baseline:
        args.baseline.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.baseline} (baseline updated)")
    elif (not args.no_compare) and args.baseline.is_file():
        b = json.loads(args.baseline.read_text())
        base_results = [
            CellResult(
                scenario=r["scenario"], buffer_size=r["buffer_size"],
                p50_us=r["p50_us"], p95_us=r["p95_us"], p99_us=r["p99_us"],
                max_us=r["max_us"], mean_us=r["mean_us"],
                block_count=r["block_count"], rtf=r["rtf"],
                deadline_us=r["deadline_us"],
                deadline_misses=r["deadline_misses"],
                deadline_miss_pct=r["deadline_miss_pct"],
            )
            for r in b["results"]
        ]
        diffs = compare(results, base_results)

    summary = format_summary(results, diffs, meta)
    args.summary.write_text(summary)
    print(f"wrote {args.summary}")
    print()
    print(summary)

    if diffs is not None:
        regressions = [d for d in diffs if d.regression]
        if regressions:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
