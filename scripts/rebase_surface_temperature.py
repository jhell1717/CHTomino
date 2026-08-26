"""Rewrite a directory of .zarr cases with surface_fields rebased to a delta
from a boundary-condition reference temperature, instead of absolute temperature.

Why: this project's normalization (model.normalization, either min_max_scaling
or mean_std_scaling) is computed GLOBALLY across all training cases (see
compute_statistics.py -> physicsnemo's compute_scaling_factors). If
case-to-case shifts in the boundary-condition temperature (e.g. Temp_inlet
varying 20+ degrees between cases) are large relative to the actual spatial
variation of temperature *within* one case's surface, that global range is
dominated by the cross-case shift -- not the spatial pattern that's actually
the interesting thing for a surrogate to predict. Two consequences, verified
against this project's actual pipeline:
  - Training loss (loss.py:loss_fn_surface, computed on globally-normalized
    values) barely penalizes getting the spatial pattern wrong, since it's a
    tiny fraction of the global range the loss is measured against.
  - The relative-L2 eval metric (utils.py:metrics_fn_surface) is
    correspondingly dominated by the (easy to get right, directly available
    from global_params_values) baseline, so it can look good even when the
    spatial pattern is essentially unlearned -- see
    l2_surf_temperature_centered, added to that function for exactly this.

Subtracting a boundary-condition-derived reference from surface_fields BEFORE
computing scaling factors and training makes the training target -- and the
global normalization range -- dominated by the spatial pattern instead. The
reference must come from a boundary condition (known for any case you'd want
to run the surrogate on), not from the target field itself (unknown for a
genuinely new case).

Usage:
    python scripts/rebase_surface_temperature.py data/train data/train_delta
    python scripts/rebase_surface_temperature.py data/val data/val_delta
    python scripts/rebase_surface_temperature.py data/test data/test_delta

Then in conf/config.yaml: point data.input_dir / data.input_dir_val /
eval.test_path at the new *_delta directories, set
data.temperature_reference_key to whatever --reference-key you used below
(so test.py can add it back for interpretable .vtp/plot output), delete any
existing scaling_factors.pkl (it was computed on absolute temperature and is
now stale), and retrain from scratch -- a checkpoint trained on absolute
temperature is not compatible with delta-temperature data or vice versa.

Does not modify the source directory; writes complete new case copies (all
arrays unchanged except surface_fields) so the original absolute-temperature
data is always still there to fall back to.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf


def rebase_case(src_path: Path, dst_path: Path, bc_names: list, reference_key: str) -> float:
    """Copy one .zarr case to dst_path with surface_fields -= that case's reference value.

    Returns the reference value used (for the printed summary).
    """
    src = zarr.open_group(str(src_path), mode="r")

    bc_values = np.asarray(src["global_params_values"]).flatten()
    if len(bc_values) != len(bc_names):
        raise ValueError(
            f"{src_path.name}: {len(bc_values)} global_params_values vs "
            f"{len(bc_names)} names in config -- can't reliably identify "
            f"'{reference_key}' for this case, refusing to guess."
        )
    reference_value = float(bc_values[bc_names.index(reference_key)])

    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst = zarr.open_group(str(dst_path), mode="w")
    for name in src.array_keys():
        array = np.asarray(src[name])
        if name == "surface_fields":
            array = (array - reference_value).astype(array.dtype)
        dst.create_array(name, shape=array.shape, dtype=array.dtype)
        dst[name][:] = array

    # Provenance only -- not in utils.get_keys_to_read's keys_to_read, so it's
    # inert to the rest of the pipeline; here purely so you can later confirm
    # what this file was rebased against without re-deriving it.
    dst.create_array("rebase_reference_value", shape=(1,), dtype=np.float32)
    dst["rebase_reference_value"][:] = np.array([reference_value], dtype=np.float32)

    return reference_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory of source *.zarr cases (absolute temperature)")
    parser.add_argument("output_dir", type=Path, help="Directory to write rebased *.zarr cases into")
    parser.add_argument(
        "--reference-key",
        default="Temp_inlet",
        help="Key in conf/config.yaml's variables.global_parameters to subtract from "
        "surface_fields (default: Temp_inlet). Pick whichever boundary condition is the "
        "actual physical reference temperature for your geometry -- this script can't "
        "determine that for you.",
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).resolve().parent.parent / "conf" / "config.yaml",
        type=Path,
        help="Path to config.yaml to read variables.global_parameters order from",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    bc_names = list(cfg.variables.global_parameters.keys())
    if args.reference_key not in bc_names:
        raise ValueError(f"--reference-key {args.reference_key!r} not in {bc_names}")

    case_paths = sorted(p for p in args.input_dir.iterdir() if p.suffix == ".zarr")
    if not case_paths:
        raise FileNotFoundError(f"No .zarr cases found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_values = []
    for case_path in case_paths:
        dst_path = args.output_dir / case_path.name
        ref_value = rebase_case(case_path, dst_path, bc_names, args.reference_key)
        reference_values.append(ref_value)
        print(f"{case_path.name}: subtracted {args.reference_key}={ref_value:.3f}")

    print(
        f"\nWrote {len(case_paths)} case(s) to {args.output_dir}. "
        f"{args.reference_key} ranged {min(reference_values):.3f}-{max(reference_values):.3f} "
        f"across these cases -- that's the scale of the baseline shift that no longer "
        f"dominates surface_fields' range in the output."
    )


if __name__ == "__main__":
    main()
