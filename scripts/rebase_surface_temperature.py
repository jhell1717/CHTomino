"""Rewrite a directory of .zarr cases with surface_fields rebased to a delta
from a boundary-condition-derived reference temperature, instead of absolute
temperature.

Why: this project's normalization (model.normalization, either min_max_scaling
or mean_std_scaling) is computed GLOBALLY across all training cases (see
compute_statistics.py -> physicsnemo's compute_scaling_factors). If
case-to-case shifts in the boundary-condition temperature are large relative
to the actual spatial variation of temperature *within* one case's surface,
that global range is dominated by the cross-case shift -- not the spatial
pattern that's actually the interesting thing for a surrogate to predict.
Two consequences, verified against this project's actual pipeline:
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
reference must come from boundary conditions (known for any case you'd want
to run the surrogate on), not from the target field itself (unknown for a
genuinely new case).

Two modes:

  --mode single (default): subtract one named boundary condition
    (--reference-key, default Temp_inlet) per case. Simple, but only correct
    if that one BC is actually the dominant driver of each case's baseline
    temperature. Check this first with
    notebooks/data_distribution_analysis.ipynb's "Which boundary condition
    actually predicts the per-case baseline?" cell -- if the best single-BC
    correlation there is well below the all-BCs R^2, single mode will leave a
    real residual baseline mismatch behind (visible as a much wider spread in
    the recomputed scaling_factors_summary.txt than you'd expect from
    within-case spatial variation alone).

  --mode linear: fits reference = intercept + sum(coefficient_i * BC_i) by
    least-squares regression of each case's mean surface_fields against its
    global_params_values, and subtracts that fitted per-case reference
    instead of a single BC. Use this if single-BC correlation is
    meaningfully weaker than the all-BCs R^2 in the notebook check above.
    Fit ONCE on your training directory (--fit-output writes the fit to a
    JSON file) and REUSE that exact fit for val/test (--fit-input reads it
    back) -- fitting separately per split would leak each split's own
    statistics into its reference and make the three splits inconsistent
    with each other and with what a real deployment (no ground truth
    available) could actually compute.

Usage (single mode):
    python scripts/rebase_surface_temperature.py data/train data/train_delta
    python scripts/rebase_surface_temperature.py data/val data/val_delta --reference-key Temp_inlet
    python scripts/rebase_surface_temperature.py data/test data/test_delta --reference-key Temp_inlet

Usage (linear mode):
    python scripts/rebase_surface_temperature.py data/train data/train_delta \\
        --mode linear --fit-output outputs/temperature_reference_fit.json
    python scripts/rebase_surface_temperature.py data/val data/val_delta \\
        --mode linear --fit-input outputs/temperature_reference_fit.json
    python scripts/rebase_surface_temperature.py data/test data/test_delta \\
        --mode linear --fit-input outputs/temperature_reference_fit.json

Then in conf/config.yaml: point data.input_dir / data.input_dir_val /
eval.test_path at the new *_delta directories; set data.temperature_reference_key
(single mode) or data.temperature_reference_fit (linear mode, path to the
JSON file) so test.py can add the reference back for interpretable
.vtp/plot output; delete any existing scaling_factors.pkl (it was computed
on absolute temperature and is now stale); and retrain from scratch -- a
checkpoint trained on absolute temperature is not compatible with
delta-temperature data or vice versa.

Does not modify the source directory; writes complete new case copies (all
arrays unchanged except surface_fields) so the original absolute-temperature
data is always still there to fall back to.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf


def read_case_bc_values(case_path: Path, bc_names: list) -> np.ndarray:
    """Read one case's global_params_values as a (len(bc_names),) array, in bc_names order."""
    group = zarr.open_group(str(case_path), mode="r")
    bc_values = np.asarray(group["global_params_values"]).flatten()
    if len(bc_values) != len(bc_names):
        raise ValueError(
            f"{case_path.name}: {len(bc_values)} global_params_values vs "
            f"{len(bc_names)} names in config -- refusing to guess a mapping."
        )
    return bc_values


def fit_linear_reference(case_paths: list, bc_names: list) -> dict:
    """Least-squares fit of each case's mean surface_fields ~ intercept + BCs.

    Returns a dict directly JSON-serializable via fit_output/fit_input.
    """
    X_rows, y_vals = [], []
    for case_path in case_paths:
        bc_values = read_case_bc_values(case_path, bc_names)
        surface_fields = np.asarray(zarr.open_group(str(case_path), mode="r")["surface_fields"])
        X_rows.append(np.append(bc_values, 1.0))
        y_vals.append(float(surface_fields.mean()))

    X = np.stack(X_rows)
    y = np.array(y_vals)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "bc_names": bc_names,
        "coefficients": {name: float(c) for name, c in zip(bc_names, coeffs[:-1])},
        "intercept": float(coeffs[-1]),
        "r_squared": r_squared,
        "num_cases_fit": len(case_paths),
    }


def apply_reference(bc_values: np.ndarray, bc_names: list, fit: dict) -> float:
    """Evaluate a fitted linear reference for one case's boundary-condition values."""
    coeffs = np.array([fit["coefficients"][name] for name in bc_names])
    return float(np.dot(coeffs, bc_values) + fit["intercept"])


def rebase_case(src_path: Path, dst_path: Path, reference_value: float) -> None:
    """Copy one .zarr case to dst_path with surface_fields -= reference_value."""
    src = zarr.open_group(str(src_path), mode="r")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory of source *.zarr cases (absolute temperature)")
    parser.add_argument("output_dir", type=Path, help="Directory to write rebased *.zarr cases into")
    parser.add_argument("--mode", choices=["single", "linear"], default="single")
    parser.add_argument(
        "--reference-key",
        default="Temp_inlet",
        help="[--mode single] Key in conf/config.yaml's variables.global_parameters to "
        "subtract from surface_fields (default: Temp_inlet).",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=None,
        help="[--mode linear] Fit the reference on input_dir's cases and save it here as JSON. "
        "Use this on your TRAINING directory only.",
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=None,
        help="[--mode linear] Reuse a previously-fit reference from this JSON file (from a "
        "--fit-output run on the training directory) instead of fitting on input_dir. Use "
        "this for val/test directories so the reference is consistent across splits and "
        "doesn't leak each split's own statistics into itself.",
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

    case_paths = sorted(p for p in args.input_dir.iterdir() if p.suffix == ".zarr")
    if not case_paths:
        raise FileNotFoundError(f"No .zarr cases found in {args.input_dir}")

    if args.mode == "single":
        if args.reference_key not in bc_names:
            raise ValueError(f"--reference-key {args.reference_key!r} not in {bc_names}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        reference_values = []
        for case_path in case_paths:
            bc_values = read_case_bc_values(case_path, bc_names)
            reference_value = float(bc_values[bc_names.index(args.reference_key)])
            rebase_case(case_path, args.output_dir / case_path.name, reference_value)
            reference_values.append(reference_value)
            print(f"{case_path.name}: subtracted {args.reference_key}={reference_value:.3f}")

        print(
            f"\nWrote {len(case_paths)} case(s) to {args.output_dir}. "
            f"{args.reference_key} ranged {min(reference_values):.3f}-{max(reference_values):.3f} "
            "across these cases."
        )
        return

    # --mode linear
    if (args.fit_output is None) == (args.fit_input is None):
        raise ValueError("--mode linear requires exactly one of --fit-output or --fit-input")

    if args.fit_output is not None:
        fit = fit_linear_reference(case_paths, bc_names)
        args.fit_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.fit_output, "w") as f:
            json.dump(fit, f, indent=2)
        print(
            f"Fit reference on {fit['num_cases_fit']} case(s) in {args.input_dir}: "
            f"R^2={fit['r_squared']:.4f}, coefficients={fit['coefficients']}, "
            f"intercept={fit['intercept']:.4f}\nSaved to {args.fit_output}"
        )
    else:
        with open(args.fit_input) as f:
            fit = json.load(f)
        if fit["bc_names"] != bc_names:
            raise ValueError(
                f"{args.fit_input} was fit against BC order {fit['bc_names']}, but "
                f"{args.config} currently declares {bc_names} -- these must match."
            )
        print(f"Reusing reference fit from {args.fit_input} (R^2={fit['r_squared']:.4f} on its training split)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_values = []
    for case_path in case_paths:
        bc_values = read_case_bc_values(case_path, bc_names)
        reference_value = apply_reference(bc_values, bc_names, fit)
        rebase_case(case_path, args.output_dir / case_path.name, reference_value)
        reference_values.append(reference_value)
        print(f"{case_path.name}: subtracted fitted reference={reference_value:.3f}")

    print(
        f"\nWrote {len(case_paths)} case(s) to {args.output_dir}. Fitted reference ranged "
        f"{min(reference_values):.3f}-{max(reference_values):.3f} across these cases."
    )


if __name__ == "__main__":
    main()
