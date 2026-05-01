#!/usr/bin/env bash
#
# Build a .clap bundle from an export staging directory.
#
# Two modes:
#   single-model:   build.sh <staging_dir> <out.clap>
#                   stages a per-model .clap from a `nablafx-export` bundle
#                   (model.onnx + plugin_meta.json).
#
#   composite TONE: build.sh tone <staging_dir> <out.clap>
#                   stages the composite TONE plugin from a
#                   `scripts/export_tone.py` bundle (tone_meta.json + 3
#                   sub-bundle dirs: auto_eq/, saturator/, la2a/).
#
# Behavior:
#   - On first run (or after cmake inputs change) configures and builds the
#     dylibs under native/clap/build/.
#   - Stages the appropriate dylib + the onnxruntime dylib + model bundle
#     resources into the .clap bundle with the standard Mac layout.
#   - Ad-hoc codesigns the bundle (`codesign --force --deep --sign -`). Local
#     dev only; distribution needs a real identity + notarize.
set -eu

usage() {
    cat <<EOF >&2
usage:
  $(basename "$0") <staging_dir> <out.clap>          # single-model
  $(basename "$0") tone <staging_dir> <out.clap>     # composite TONE
EOF
    exit 2
}

MODE="single"
if [ "${1:-}" = "tone" ]; then
    MODE="tone"
    shift
fi

if [ "$#" -ne 2 ]; then
    usage
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build.sh: this step must run on macOS (arm64)" >&2
    exit 1
fi

STAGING="$(cd "$1" && pwd)"
OUT="$2"

if [ "$MODE" = "single" ]; then
    if [ ! -f "$STAGING/model.onnx" ] || [ ! -f "$STAGING/plugin_meta.json" ]; then
        echo "error: $STAGING is missing model.onnx or plugin_meta.json" >&2
        exit 1
    fi
else
    if [ ! -f "$STAGING/tone_meta.json" ]; then
        echo "error: $STAGING is missing tone_meta.json" >&2
        exit 1
    fi
    for sub in saturator la2a; do
        if [ ! -d "$STAGING/$sub" ]; then
            echo "error: $STAGING is missing the $sub/ sub-bundle" >&2
            exit 1
        fi
    done
    # auto_eq is now multi-class: one auto_eq_<class>/ dir per preset.
    # tone_meta.json's auto_eq.classes map names which dirs to copy.
    AUTOEQ_DIRS=$(/usr/bin/env python3 - "$STAGING/tone_meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
classes = m.get("auto_eq", {}).get("classes") or {}
print(" ".join(sorted(set(classes.values()))))
PY
)
    if [ -z "${AUTOEQ_DIRS}" ]; then
        echo "error: tone_meta.json has no auto_eq.classes map" >&2
        exit 1
    fi
    for dir in $AUTOEQ_DIRS; do
        if [ ! -d "$STAGING/$dir" ]; then
            echo "error: $STAGING is missing auto_eq sub-bundle $dir/" >&2
            exit 1
        fi
    done
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$HERE/build"

# Configure if needed; always build so source changes are picked up.
NEED_CONFIGURE=0
[ -f "$BUILD_DIR/build_config.sh" ] || NEED_CONFIGURE=1
[ -f "$BUILD_DIR/nablafx_clap.so" ] || NEED_CONFIGURE=1
[ -f "$BUILD_DIR/tone_clap.so"    ] || NEED_CONFIGURE=1
if [ "$NEED_CONFIGURE" -eq 1 ]; then
    cmake -S "$HERE" -B "$BUILD_DIR" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$BUILD_DIR" -j

# shellcheck disable=SC1091
. "$BUILD_DIR/build_config.sh"

if [ "$MODE" = "single" ]; then
    META_PATH="$STAGING/plugin_meta.json"
    DYLIB_PATH="$NABLAFX_CLAP_DYLIB"
else
    META_PATH="$STAGING/tone_meta.json"
    DYLIB_PATH="$TONE_CLAP_DYLIB"
fi

pyscript='
import json, re, sys
m = json.load(open(sys.argv[1]))
print(m["model_id"])
print(m["effect_name"])
# Sanitize effect_name for use as a filename (model_id can exceed 255-char OS limit)
exe = re.sub(r"[^a-zA-Z0-9_-]", "_", m["effect_name"]).lower().strip("_")
print(exe)
'
PYOUT=$(/usr/bin/env python3 -c "$pyscript" "$META_PATH")
MODEL_ID=$(printf '%s\n' "$PYOUT" | sed -n '1p')
EFFECT_NAME=$(printf '%s\n' "$PYOUT" | sed -n '2p')
EXECUTABLE=$(printf '%s\n' "$PYOUT" | sed -n '3p')

# Layout:
#   <OUT>/Contents/Info.plist
#   <OUT>/Contents/MacOS/<executable>
#   <OUT>/Contents/Frameworks/libonnxruntime.<ver>.dylib
#   <OUT>/Contents/Frameworks/libonnxruntime.dylib (symlink)
#   <OUT>/Contents/Resources/...      (model.onnx + plugin_meta.json for single,
#                                       tone_meta.json + 3 sub-bundle dirs for tone)
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Frameworks" "$OUT/Contents/Resources"

cp "$DYLIB_PATH"             "$OUT/Contents/MacOS/$EXECUTABLE"
cp "$NABLAFX_CLAP_ORT_DYLIB" "$OUT/Contents/Frameworks/"
ln -sf "$(basename "$NABLAFX_CLAP_ORT_DYLIB")" \
       "$OUT/Contents/Frameworks/libonnxruntime.dylib"

if [ "$MODE" = "single" ]; then
    cp "$STAGING/model.onnx"        "$OUT/Contents/Resources/"
    cp "$STAGING/plugin_meta.json"  "$OUT/Contents/Resources/"
else
    cp "$STAGING/tone_meta.json"    "$OUT/Contents/Resources/"
    for sub in saturator la2a; do
        cp -R "$STAGING/$sub" "$OUT/Contents/Resources/"
    done
    for dir in $AUTOEQ_DIRS; do
        cp -R "$STAGING/$dir" "$OUT/Contents/Resources/"
    done
    # Copy the WebUI so the plugin can load it at runtime.
    cp -R "$HERE/ui" "$OUT/Contents/Resources/"
fi

BUNDLE_ID="com.nablafx.$EXECUTABLE"
sed \
    -e "s|__BUNDLE_EXECUTABLE__|$EXECUTABLE|g" \
    -e "s|__BUNDLE_IDENTIFIER__|$BUNDLE_ID|g" \
    -e "s|__BUNDLE_NAME__|$EFFECT_NAME|g" \
    "$HERE/template/Info.plist.in" > "$OUT/Contents/Info.plist"

# Ad-hoc codesign for local dev. Distribution builds need a real identity.
codesign --force --deep --sign - "$OUT"

echo "built $OUT"
echo "  mode:        $MODE"
echo "  effect_name: $EFFECT_NAME"
echo "  model_id:    $MODEL_ID"
echo "  bundle_id:   $BUNDLE_ID"
