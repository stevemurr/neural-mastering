"""Build the composite NeuralMastering staging bundle.

Three modes:

  Bundles already exported (the typical Mac-clone-and-build path)::

      python scripts/export_tone.py from-bundles \\
          --auto-eq-bass-bundle      artifacts/tone-bundles/auto_eq_bass \\
          --auto-eq-drums-bundle     artifacts/tone-bundles/auto_eq_drums \\
          --auto-eq-vocals-bundle    artifacts/tone-bundles/auto_eq_vocals \\
          --auto-eq-other-bundle     artifacts/tone-bundles/auto_eq_other \\
          --auto-eq-full-mix-bundle  artifacts/tone-bundles/auto_eq_full_mix \\
          --saturator-bundle         artifacts/tone-bundles/saturator \\
          --la2a-bundle              artifacts/tone-bundles/la2a \\
          --out                      build/tone-staging

  From-runs (export each stage from a Hydra run dir, then compose). Each
  ``--auto-eq-<class>-run`` is optional — the composite needs at least one
  class but accepts any subset; ``--default-class`` must be one of the classes
  you provide::

      python scripts/export_tone.py from-runs \\
          --auto-eq-full-mix-run /shared/artifacts/auto_eq_musdb_full_mix/.../<ts> \\
          --auto-eq-bass-run     /shared/artifacts/auto_eq_musdb_bass/.../<ts> \\
          --saturator-run        /shared/artifacts/saturator_synth/.../<ts> \\
          --la2a-run             /shared/artifacts/la2a/.../<ts> \\
          --la2a-ckpt            /shared/artifacts/la2a/.../checkpoints/epoch=8-step=89600.ckpt \\
          --default-class        full_mix \\
          --out                  build/tone-staging

  From-class-dir (auto-discover the latest checkpoint under each
  ``/shared/artifacts/auto_eq_musdb_<class>/outputs/<date>/<time>/``)::

      python scripts/export_tone.py from-class-dir \\
          --auto-eq-root /shared/artifacts \\
          --saturator-run /shared/artifacts/saturator_synth/.../<ts> \\
          --la2a-run      /shared/artifacts/la2a/.../<ts> \\
          --la2a-ckpt     <best_la2a_ckpt> \\
          --out           build/tone-staging

The ``from-runs`` and ``from-class-dir`` modes shell out to ``nablafx-export``
for each stage so the same export code path runs as you'd get from manual
exports.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import nablafx  # noqa: F401 — applies rational-activations patch
from neural_mastering.export.composite import (
    DEFAULT_ACTIVE_CLASS,
    DEFAULT_CLASS_ORDER,
    export_composite_bundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _collect_autoeq_runs(args) -> Dict[str, str]:
    """Pull all --auto-eq-<class>-run flags from args into a dict."""
    out: Dict[str, str] = {}
    for cls in DEFAULT_CLASS_ORDER:
        v = getattr(args, f"auto_eq_{cls}_run", None)
        if v:
            out[cls] = v
    return out


def _collect_autoeq_ckpts(args) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for cls in DEFAULT_CLASS_ORDER:
        out[cls] = getattr(args, f"auto_eq_{cls}_ckpt", None)
    return out


def _collect_autoeq_bundles(args) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cls in DEFAULT_CLASS_ORDER:
        v = getattr(args, f"auto_eq_{cls}_bundle", None)
        if v:
            out[cls] = v
    return out


def _autoeq_bundles_from_runs(autoeq_runs: Dict[str, str],
                              autoeq_ckpts: Dict[str, Optional[str]],
                              work: Path) -> Dict[str, Path]:
    bundles: Dict[str, Path] = {}
    for cls, run in autoeq_runs.items():
        out = work / f"auto_eq_{cls}"
        cmd = ["nablafx-export", "--run-dir", run]
        ckpt = autoeq_ckpts.get(cls)
        if ckpt:
            cmd += ["--ckpt", ckpt]
        cmd += ["--out", str(out)]
        _run(cmd)
        bundles[cls] = out
    return bundles


def _print_summary(args, meta) -> None:
    print(f"composite bundle written to {args.out}")
    print(f"  effect_name:   {meta.effect_name}")
    print(f"  model_id:      {meta.model_id}")
    print(f"  sample_rate:   {meta.sample_rate}")
    print(f"  classes:       {', '.join(meta.auto_eq.get('class_order', []))}")
    print(f"  default_class: {meta.auto_eq.get('default_class')}")


# ---------------------------------------------------------------------------
# Mode entrypoints
# ---------------------------------------------------------------------------


def _from_bundles(args) -> int:
    autoeq_bundles = _collect_autoeq_bundles(args)
    if not autoeq_bundles:
        print("error: at least one --auto-eq-<class>-bundle is required",
              file=sys.stderr)
        return 2
    meta = export_composite_bundle(
        auto_eq_bundles={c: Path(p) for c, p in autoeq_bundles.items()},
        saturator_bundle=Path(args.saturator_bundle),
        la2a_bundle=Path(args.la2a_bundle),
        out_dir=Path(args.out),
        effect_name=args.effect_name,
        default_class=args.default_class,
    )
    _print_summary(args, meta)
    return 0


def _from_runs(args) -> int:
    autoeq_runs = _collect_autoeq_runs(args)
    if not autoeq_runs:
        print("error: at least one --auto-eq-<class>-run is required",
              file=sys.stderr)
        return 2
    autoeq_ckpts = _collect_autoeq_ckpts(args)

    work = Path(tempfile.mkdtemp(prefix="tone-export-"))
    try:
        sat_dir  = work / "saturator"
        la2a_dir = work / "la2a"

        _run([
            "nablafx-export", "--run-dir", args.saturator_run,
            *(["--ckpt", args.saturator_ckpt] if args.saturator_ckpt else []),
            "--out", str(sat_dir),
        ])
        _run([
            "nablafx-export", "--run-dir", args.la2a_run,
            *(["--ckpt", args.la2a_ckpt] if args.la2a_ckpt else []),
            "--effect", "LA2A", "--letters", "C,P",
            "--out", str(la2a_dir),
        ])
        autoeq_bundles = _autoeq_bundles_from_runs(autoeq_runs, autoeq_ckpts, work)

        meta = export_composite_bundle(
            auto_eq_bundles=autoeq_bundles,
            saturator_bundle=sat_dir,
            la2a_bundle=la2a_dir,
            out_dir=Path(args.out),
            effect_name=args.effect_name,
            default_class=args.default_class,
        )
        _print_summary(args, meta)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _latest_run_dir(class_root: Path) -> Optional[Path]:
    """Pick the lexicographically-latest <date>/<time>/ under
    ``class_root/outputs/`` (Hydra's directory layout)."""
    outputs = class_root / "outputs"
    if not outputs.is_dir():
        return None
    candidates: List[Path] = []
    for date_dir in outputs.iterdir():
        if not date_dir.is_dir():
            continue
        for time_dir in date_dir.iterdir():
            if (time_dir / ".hydra").is_dir():
                candidates.append(time_dir)
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _from_class_dir(args) -> int:
    auto_root = Path(args.auto_eq_root).resolve()
    autoeq_runs: Dict[str, str] = {}
    for cls in DEFAULT_CLASS_ORDER:
        cls_root = auto_root / f"auto_eq_musdb_{cls}"
        run = _latest_run_dir(cls_root)
        if run is None:
            print(f"warning: no Hydra run found under {cls_root}/outputs",
                  file=sys.stderr)
            continue
        autoeq_runs[cls] = str(run)
        print(f"  picked {cls}: {run}", flush=True)
    if not autoeq_runs:
        print(f"error: no auto-EQ runs discovered under {auto_root}",
              file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="tone-export-"))
    try:
        sat_dir  = work / "saturator"
        la2a_dir = work / "la2a"
        _run([
            "nablafx-export", "--run-dir", args.saturator_run,
            *(["--ckpt", args.saturator_ckpt] if args.saturator_ckpt else []),
            "--out", str(sat_dir),
        ])
        _run([
            "nablafx-export", "--run-dir", args.la2a_run,
            *(["--ckpt", args.la2a_ckpt] if args.la2a_ckpt else []),
            "--effect", "LA2A", "--letters", "C,P",
            "--out", str(la2a_dir),
        ])
        autoeq_bundles = _autoeq_bundles_from_runs(
            autoeq_runs, {c: None for c in autoeq_runs}, work
        )
        meta = export_composite_bundle(
            auto_eq_bundles=autoeq_bundles,
            saturator_bundle=sat_dir,
            la2a_bundle=la2a_dir,
            out_dir=Path(args.out),
            effect_name=args.effect_name,
            default_class=args.default_class,
        )
        _print_summary(args, meta)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_class_run_args(parser: argparse.ArgumentParser, kind: str) -> None:
    """``kind`` is "run" or "ckpt" or "bundle" — appears in the flag suffix."""
    for cls in DEFAULT_CLASS_ORDER:
        flag = f"--auto-eq-{cls.replace('_', '-')}-{kind}"
        parser.add_argument(flag, default=None, dest=f"auto_eq_{cls}_{kind}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args available on every subparser AND the top-level parser, so callers
    can put them before or after the subcommand without surprise."""
    parser.add_argument("--effect-name", default="NeuralMastering")
    parser.add_argument("--default-class", default=DEFAULT_ACTIVE_CLASS,
                        help=f"Class loaded by default in the plugin "
                             f"(one of {','.join(DEFAULT_CLASS_ORDER)}).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="export_tone",
                                description="Build composite NeuralMastering staging bundle.")
    _add_common_args(p)
    sp = p.add_subparsers(dest="cmd", required=True)

    pb = sp.add_parser("from-bundles",
                       help="compose pre-exported per-stage bundles")
    _add_common_args(pb)
    _add_class_run_args(pb, "bundle")
    pb.add_argument("--saturator-bundle", required=True)
    pb.add_argument("--la2a-bundle",      required=True)
    pb.add_argument("--out",              required=True)
    pb.set_defaults(func=_from_bundles)

    pr = sp.add_parser("from-runs",
                       help="export each stage from its Hydra run dir, then compose")
    _add_common_args(pr)
    _add_class_run_args(pr, "run")
    _add_class_run_args(pr, "ckpt")
    pr.add_argument("--saturator-run", required=True)
    pr.add_argument("--saturator-ckpt", default=None)
    pr.add_argument("--la2a-run",      required=True)
    pr.add_argument("--la2a-ckpt",     default=None)
    pr.add_argument("--out",           required=True)
    pr.set_defaults(func=_from_runs)

    pc = sp.add_parser("from-class-dir",
                       help="auto-discover the latest run for each class under "
                            "<root>/auto_eq_musdb_<class>/outputs/")
    _add_common_args(pc)
    pc.add_argument("--auto-eq-root", required=True,
                    help="Parent dir holding auto_eq_musdb_<class>/ subdirs.")
    pc.add_argument("--saturator-run", required=True)
    pc.add_argument("--saturator-ckpt", default=None)
    pc.add_argument("--la2a-run",      required=True)
    pc.add_argument("--la2a-ckpt",     default=None)
    pc.add_argument("--out",           required=True)
    pc.set_defaults(func=_from_class_dir)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
