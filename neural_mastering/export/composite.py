"""Composite NeuralMastering plugin export.

Reads pre-built per-stage bundles (la2a, saturator, and N per-class auto_eq —
all produced by ``nablafx-export``) and emits a single staging directory ready
for ``native/clap/build.sh tone`` on macOS.

Composite layout written to ``out_dir``::

    tone_meta.json                # this module's schema; see CompositePluginMeta
    la2a/                         (copied from input bundle)
    saturator/                    (no model.onnx — pure DSP stage)
    auto_eq_<class>/              (N copies, one per class — controller LSTM ONNX
                                  + identical PEQ DSP block in plugin_meta.json)

The C++ side (``native/clap/src/composite_meta.cpp``) loads ``tone_meta.json``,
the saturator + la2a sub-``plugin_meta.json``, and one auto_eq sub-meta per
class. All auto_eq classes must share the same PEQ DSP layout (frozen freqs
+ identical ranges) so the runtime can swap which controller ONNX is active
without changing the downstream biquad cascade.

Schema versions:
    1 — single-class auto_eq (deprecated; brown-noise direction)
    2 — multi-class auto_eq with per-class controller bundles + CLS control
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _AmountMappingSat:
    pre_gain_db_max:  float = 12.0
    post_gain_db_max: float = -12.0
    wet_mix_max:      float = 1.0


@dataclass(frozen=True)
class _AmountMappingLa2a:
    peak_reduction_min: float = 20.0
    peak_reduction_max: float = 70.0
    comp_or_limit:      float = 1.0   # held at "Limit" in the composite spec


@dataclass(frozen=True)
class _AmountMappingAutoEq:
    wet_mix_max: float = 1.0


@dataclass(frozen=True)
class _AmountMappingSslComp:
    wet_mix_max: float = 1.0   # SSC knob → ssl_comp wet/dry mix


@dataclass(frozen=True)
class CompositePluginMeta:
    """Top-level meta for the composite NeuralMastering plugin.

    The C++ host reads this once at module load to wire AMT → per-stage params,
    locate sub-bundles, and configure the in-host DSP stages (LUFS leveler,
    true-peak ceiling, output trim).
    """
    schema_version: int = SCHEMA_VERSION
    effect_name:    str = "NeuralMastering"
    model_id:       str = ""
    sample_rate:    int = 44100
    channels:       int = 1
    # Single-instance sub-bundles (saturator, la2a). Directory names relative to
    # the staging dir / the .clap Resources dir.
    sub_bundles:    Dict[str, str] = field(default_factory=dict)
    # Multi-class auto-EQ — one bundle per instrument-class preset.
    auto_eq:        Dict[str, Any] = field(default_factory=dict)
    # AMT (Amount), TRM (Output Trim), CLS (auto-EQ class) etc.
    controls:       Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-stage mapping from AMT ∈ [0, 1] to internal parameters.
    amount_mapping: Dict[str, Dict[str, float]] = field(default_factory=dict)
    leveler:        Dict[str, float] = field(default_factory=dict)
    ceiling:        Dict[str, float] = field(default_factory=dict)


# Stable canonical class order; bundle layout uses this for `min`/`max` of
# the CLS control and matches the C++ side's index-to-class lookup.
DEFAULT_CLASS_ORDER: Tuple[str, ...] = ("bass", "drums", "vocals", "other", "full_mix")
DEFAULT_ACTIVE_CLASS: str = "full_mix"


def _build_default_meta(
    model_id: str,
    sample_rate: int,
    classes: Iterable[str],
    default_class: str,
) -> CompositePluginMeta:
    classes = list(classes)
    if default_class not in classes:
        raise ValueError(f"default_class {default_class!r} not in classes {classes!r}")
    n_classes = len(classes)
    default_idx = classes.index(default_class)
    auto_eq = {
        "default_class": default_class,
        "classes":       {c: f"auto_eq_{c}" for c in classes},
        # Stable ordering — the CLS control's integer value indexes into this.
        "class_order":   list(classes),
    }
    controls = {
        "LVL": {"id": "LVL", "name": "Leveler",     "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "LVT": {"id": "LVT", "name": "Lev Target",  "min": -36.0, "max": -6.0,
                "default": -14.0, "skew": 1.0, "unit": "LUFS"},
        "SDR": {"id": "SDR", "name": "Sat Drive",   "min": 0.0,   "max": 24.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "SVO": {"id": "SVO", "name": "Sat Output",  "min": -24.0, "max": 12.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "SMX": {"id": "SMX", "name": "Sat Mix",     "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "SHF": {"id": "SHF", "name": "Sat HPF",    "min": 20.0,  "max": 500.0,
                "default": 20.0, "skew": 1.0, "unit": "Hz"},
        "STH": {"id": "STH", "name": "Sat Thresh", "min": -24.0, "max": 0.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "SBS": {"id": "SBS", "name": "Sat Bias",   "min": -0.5,  "max": 0.5,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "C_L": {"id": "C_L", "name": "Comp/Limit",     "min": 0.0,   "max": 1.0,
                "default": 1.0, "skew": 1.0, "unit": "switch"},
        "CMP": {"id": "CMP", "name": "Peak Reduction", "min": 0.0,   "max": 100.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "SSC": {"id": "SSC", "name": "Bus Comp",       "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "CLS": {"id": "CLS", "name": "EQ Class",
                "min": 0.0, "max": float(max(0, n_classes - 1)),
                "default": float(default_idx), "skew": 1.0, "unit": "enum"},
        "EQ":  {"id": "EQ",  "name": "Auto EQ",     "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "EQR": {"id": "EQR", "name": "EQ Range",    "min": 0.0,   "max": 1.0,
                "default": 1.0, "skew": 1.0, "unit": ""},
        "EQS": {"id": "EQS", "name": "EQ Speed",    "min": 10.0,  "max": 500.0,
                "default": 100.0, "skew": 1.0, "unit": "ms"},
        "EQ0": {"id": "EQ0", "name": "EQ Low Shelf","min": -9.0,  "max": 9.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "EQ1": {"id": "EQ1", "name": "EQ 110 Hz",   "min": -9.0,  "max": 9.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "EQ2": {"id": "EQ2", "name": "EQ 1.1 kHz",  "min": -9.0,  "max": 9.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "EQ3": {"id": "EQ3", "name": "EQ 7 kHz",    "min": -9.0,  "max": 9.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "EQ4": {"id": "EQ4", "name": "EQ High Shelf","min": -9.0, "max": 9.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
        "OLV": {"id": "OLV", "name": "Out Leveler",  "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "OLT": {"id": "OLT", "name": "Out Lev Target", "min": -36.0, "max": -6.0,
                "default": -14.0, "skew": 1.0, "unit": "LUFS"},
        "SPW": {"id": "SPW", "name": "Spatial Wet",   "min": 0.0,   "max": 1.0,
                "default": 0.0, "skew": 1.0, "unit": ""},
        "SPR": {"id": "SPR", "name": "Spatial Rate",  "min": 0.1,   "max": 2.0,
                "default": 0.8, "skew": 1.0, "unit": "Hz"},
        "SPD": {"id": "SPD", "name": "Spatial Depth", "min": 0.0,   "max": 1.0,
                "default": 0.5, "skew": 1.0, "unit": ""},
        "TRM": {"id": "TRM", "name": "Output Trim", "min": -12.0, "max": 12.0,
                "default": 0.0, "skew": 1.0, "unit": "dB"},
    }
    return CompositePluginMeta(
        model_id=model_id,
        sample_rate=sample_rate,
        sub_bundles={
            "saturator": "saturator",
            "la2a":      "la2a",
            "ssl_comp":  "ssl_comp",
        },
        auto_eq=auto_eq,
        controls=controls,
        amount_mapping={
            "saturator": asdict(_AmountMappingSat()),
            "la2a":      asdict(_AmountMappingLa2a()),
            "auto_eq":   asdict(_AmountMappingAutoEq()),
            "ssl_comp":  asdict(_AmountMappingSslComp()),
        },
        leveler={"target_lufs": -14.0},
        ceiling={"ceiling_dbtp": -1.0, "lookahead_ms": 1.5,
                 "attack_ms": 0.5, "release_ms": 50.0},
    )


def _load_sub_meta(bundle_dir: Path) -> Dict[str, Any]:
    p = bundle_dir / "plugin_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text())


def _check_sub_bundle(bundle_dir: Path, expected_kind: str, expected_block_kind: Optional[str] = None) -> Dict[str, Any]:
    meta = _load_sub_meta(bundle_dir)
    sk = meta.get("stage_kind")
    if sk != expected_kind:
        raise ValueError(f"{bundle_dir}: expected stage_kind={expected_kind!r}, got {sk!r}")
    if expected_block_kind:
        blocks = meta.get("dsp_blocks") or []
        if not blocks or blocks[0].get("kind") != expected_block_kind:
            raise ValueError(
                f"{bundle_dir}: expected dsp_blocks[0].kind={expected_block_kind!r}, "
                f"got {[b.get('kind') for b in blocks]}"
            )
    return meta


def export_composite_bundle(
    auto_eq_bundles: Mapping[str, Path],
    saturator_bundle: Path,
    la2a_bundle: Path,
    out_dir: Path,
    effect_name: str = "NeuralMastering",
    default_class: str = DEFAULT_ACTIVE_CLASS,
    class_order: Optional[List[str]] = None,
) -> CompositePluginMeta:
    """Validate sub-bundles, copy them under ``out_dir``, and write
    ``tone_meta.json``.

    ``auto_eq_bundles`` is a mapping of class name → directory holding a
    ``nablafx-export`` bundle for that class's controller+DSP. All classes must
    share the same SpectralMaskEQ geometry.
    """
    if not auto_eq_bundles:
        raise ValueError("auto_eq_bundles must contain at least one class")

    # Resolve class order: caller-provided > canonical (filtered to bundles given)
    # > insertion order from the dict.
    if class_order is None:
        ordered = [c for c in DEFAULT_CLASS_ORDER if c in auto_eq_bundles]
        # Append any extra classes not in the canonical list, preserving caller order.
        for c in auto_eq_bundles.keys():
            if c not in ordered:
                ordered.append(c)
    else:
        for c in class_order:
            if c not in auto_eq_bundles:
                raise ValueError(f"class_order entry {c!r} not in auto_eq_bundles")
        for c in auto_eq_bundles.keys():
            if c not in class_order:
                raise ValueError(f"auto_eq_bundles key {c!r} missing from class_order")
        ordered = list(class_order)

    if default_class not in ordered:
        raise ValueError(f"default_class {default_class!r} not in classes {ordered!r}")

    auto_eq_paths = {c: Path(auto_eq_bundles[c]).resolve() for c in ordered}
    saturator_bundle = Path(saturator_bundle).resolve()
    la2a_bundle      = Path(la2a_bundle).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate every auto-EQ sub-bundle. All classes must declare
    # spectral_mask_eq with identical geometry so the runtime can dispatch
    # without per-class layout assertions.
    autoeq_metas: Dict[str, Dict[str, Any]] = {}
    canonical_sig: Optional[Tuple] = None
    canonical_cls: Optional[str] = None
    for cls, p in auto_eq_paths.items():
        m = _check_sub_bundle(p, expected_kind="nn+dsp",
                              expected_block_kind="spectral_mask_eq")
        blocks = m.get("dsp_blocks") or []
        p_ = blocks[0].get("params", {})
        sig = (
            p_.get("sample_rate"), p_.get("block_size"),
            p_.get("num_control_params"), p_.get("n_fft"),
            p_.get("hop"), p_.get("n_bands"),
            p_.get("min_gain_db"), p_.get("max_gain_db"),
            p_.get("f_min"), p_.get("f_max"),
        )
        if canonical_sig is None:
            canonical_sig = sig
            canonical_cls = cls
        elif sig != canonical_sig:
            raise ValueError(
                f"auto_eq class {cls!r} layout differs from "
                f"{canonical_cls!r}; classes must share geometry."
            )
        autoeq_metas[cls] = m

    sat_meta  = _check_sub_bundle(saturator_bundle, expected_kind="dsp",
                                  expected_block_kind="rational_a")
    la2a_meta = _check_sub_bundle(la2a_bundle,      expected_kind="nn")

    # Sample rate must be uniform across every stage of the chain.
    sample_rates = {sat_meta["sample_rate"], la2a_meta["sample_rate"]}
    sample_rates.update(m["sample_rate"] for m in autoeq_metas.values())
    if len(sample_rates) != 1:
        raise ValueError(
            "sub-bundles disagree on sample_rate: "
            f"saturator={sat_meta['sample_rate']}, la2a={la2a_meta['sample_rate']}, "
            "auto_eq=" + ", ".join(f"{c}={m['sample_rate']}"
                                   for c, m in autoeq_metas.items())
        )
    sample_rate = sample_rates.pop()

    # Compose a stable model_id from the chain. Auto-EQ contributes the
    # joined per-class model_ids so DAWs reload automation against the exact
    # combination shipped.
    autoeq_id = "_".join(autoeq_metas[c]["model_id"] for c in ordered)
    model_id = f"nm__{la2a_meta['model_id']}__{sat_meta['model_id']}__{autoeq_id}"

    # Copy sub-bundles into the staging dir under their stable role names.
    for role, src in (("saturator", saturator_bundle), ("la2a", la2a_bundle)):
        dst = out_dir / role
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for cls, src in auto_eq_paths.items():
        dst = out_dir / f"auto_eq_{cls}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    meta = _build_default_meta(model_id=model_id, sample_rate=int(sample_rate),
                               classes=ordered, default_class=default_class)
    meta = CompositePluginMeta(
        **{**asdict(meta), "effect_name": effect_name},
    )
    (out_dir / "tone_meta.json").write_text(json.dumps(asdict(meta), indent=2) + "\n")
    return meta
