#!/usr/bin/env bash
#
# Turnkey installer for the composite NeuralMastering CLAP plugin on macOS (arm64).
#
# Usage:
#   scripts/install_tone_mac.sh [--bundle DIR] [--out DIR] [--no-install]
#
# What it does:
#   1. Locates a pre-built composite staging directory containing
#      tone_meta.json + saturator/ + la2a/ + auto_eq_<class>/ sub-bundle
#      directories. Lookup order:
#         1. --bundle DIR (CLI override)
#         2. ./weights/tone_bundle/   (git-tracked, populated by training host)
#         3. ./build/tone-staging/    (local rebuild output)
#         4. ./artifacts/tone-bundle/ (legacy CI drop path)
#      If none exist, errors with a hint to run scripts/export_tone.py first
#      (only meaningful on the training host where checkpoints live).
#   2. Builds the tone_clap dylib via native/clap/build.sh tone, packaging into
#      $OUT (default: build/NeuralMastering.clap).
#   3. Unless --no-install is passed, copies the .clap bundle to
#      ~/Library/Audio/Plug-Ins/CLAP/ so DAWs can pick it up.
#
# This is the only script a Mac user needs to clone-and-run after pulling the
# repo. The staging dir should be checked in or rsynced from the training host
# (it contains the ONNX models — too big for git typically; rsync from
# /shared/artifacts/exports/tone-staging/ on the GPU box is the canonical
# path).
set -eu

usage() {
    cat <<EOF >&2
usage: $(basename "$0") [--bundle <staging_dir>] [--out <out.clap>] [--no-install]

  --bundle DIR    Composite staging dir (default: ./build/tone-staging or
                  ./artifacts/tone-bundle, whichever exists).
  --out PATH      Output .clap bundle path (default: ./build/NeuralMastering.clap).
  --no-install    Build only; don't copy to ~/Library/Audio/Plug-Ins/CLAP/.
EOF
    exit 2
}

if [ "$(uname -s)" != "Darwin" ]; then
    echo "install_tone_mac.sh: must run on macOS (arm64)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE=""
OUT="$REPO_ROOT/build/NeuralMastering.clap"
DO_INSTALL=1

while [ $# -gt 0 ]; do
    case "$1" in
        --bundle)     BUNDLE="$2"; shift 2 ;;
        --out)        OUT="$2";    shift 2 ;;
        --no-install) DO_INSTALL=0; shift ;;
        -h|--help)    usage ;;
        *)            echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [ -z "$BUNDLE" ]; then
    if   [ -d "$REPO_ROOT/weights/tone_bundle"    ]; then BUNDLE="$REPO_ROOT/weights/tone_bundle"
    elif [ -d "$REPO_ROOT/build/tone-staging"     ]; then BUNDLE="$REPO_ROOT/build/tone-staging"
    elif [ -d "$REPO_ROOT/artifacts/tone-bundle"  ]; then BUNDLE="$REPO_ROOT/artifacts/tone-bundle"
    else
        cat <<EOF >&2
error: no composite staging dir found.

Expected one of:
  $REPO_ROOT/weights/tone_bundle/      (committed in this repo)
  $REPO_ROOT/build/tone-staging/       (local export output)
  $REPO_ROOT/artifacts/tone-bundle/    (legacy CI drop)

To produce one (typically on the training host, not your Mac):

  python scripts/export_tone.py from-class-dir \\
      --auto-eq-root /shared/artifacts \\
      --saturator-run /shared/artifacts/saturator_synth/.../<ts> \\
      --la2a-run      /shared/artifacts/la2a/.../<ts> \\
      --la2a-ckpt     /shared/artifacts/la2a/.../<best.ckpt> \\
      --out           weights/tone_bundle

Then commit weights/tone_bundle/ and pull on the Mac.
EOF
        exit 1
    fi
fi

if [ ! -f "$BUNDLE/tone_meta.json" ]; then
    echo "error: $BUNDLE is missing tone_meta.json" >&2
    exit 1
fi

echo "[install_tone_mac] building NeuralMastering.clap"
echo "  staging: $BUNDLE"
echo "  out:     $OUT"

bash "$REPO_ROOT/native/clap/build.sh" tone "$BUNDLE" "$OUT"

if [ "$DO_INSTALL" -eq 1 ]; then
    INSTALL_DIR="$HOME/Library/Audio/Plug-Ins/CLAP"
    mkdir -p "$INSTALL_DIR"
    INSTALLED="$INSTALL_DIR/$(basename "$OUT")"
    rm -rf "$INSTALLED"
    cp -R "$OUT" "$INSTALLED"
    echo "[install_tone_mac] installed to $INSTALLED"
    echo
    echo "Done. Restart your DAW (or rescan plug-ins) and look for NeuralMastering."
else
    echo "[install_tone_mac] build complete (skipped install per --no-install)"
fi
