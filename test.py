"""Evaluate the trained surface-temperature DoMINO model on a held-out set of
.zarr test cases (same layout as the training data -- see
ref_material/zarr_structure.md).

For each case this writes:
  - a point-cloud .vtp with TemperaturePred / TemperatureTrue point data
    (physical, unnormalized units)
  - a predicted-vs-true parity plot (PNG)
and prints the per-case and overall mean relative L2 error.

To evaluate a checkpoint from a different run than the live conf/config.yaml
(e.g. a historical architecture/normalization), override on the command line:
    python test.py eval.model_config=/path/to/old_run/resolved_config.yaml
When eval.model_config is set, cfg.model / cfg.variables / cfg.data (the
things the checkpoint was actually trained with) come from that historical
resolved_config.yaml; only eval.* itself (where to find test data, where to
save predictions) still comes from the live config passed to this script.
"""

import json
import os
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pyvista as pv
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")  # headless-safe: no X server on training/eval nodes
import matplotlib.pyplot as plt

from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import create_directory, unnormalize

import cuml_knn_patch  # noqa: F401  -- see that file: works around a RAPIDS/physicsnemo cuML API mismatch
import geometry_sampling_patch  # noqa: F401  -- see that file: area-weights geom_points_sample instead of uniform-per-vertex
from utils import assert_surface_only, compute_l2, get_keys_to_read, get_num_vars, load_scaling_factors


def resolve_model_and_checkpoint(cfg: DictConfig) -> tuple[DictConfig, str]:
    """Resolve which model config + checkpoint directory to evaluate.

    By default (eval.model_config unset), uses the live hydra config and
    this run's own resume_dir/best_model. See module docstring for the
    historical-run override.
    """
    model_config_path = cfg.eval.get("model_config", None)
    checkpoint_dir_override = cfg.eval.get("model_checkpoint_dir", None)

    if model_config_path is None:
        checkpoint_dir = checkpoint_dir_override or os.path.join(cfg.resume_dir, "best_model")
        return cfg, to_absolute_path(checkpoint_dir)

    model_config_path = to_absolute_path(model_config_path)
    model_cfg = OmegaConf.load(model_config_path)
    print(f"Loaded historical model config from '{model_config_path}'")

    if checkpoint_dir_override:
        checkpoint_dir = to_absolute_path(checkpoint_dir_override)
    else:
        checkpoint_dir = os.path.join(os.path.dirname(model_config_path), "models", "best_model")

    return model_cfg, checkpoint_dir


def plot_pred_vs_true(pred, true, var_names, save_path, title_prefix="", max_points=200_000):
    """Save a predicted-vs-actual parity scatter plot to `save_path`.

    One subplot per surface variable (columns of `pred`/`true`), each showing
    predicted value (y) against ground-truth value (x), a y=x reference line,
    and the relative L2 error for that variable in the title.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.ndim == 1:
        true = true[:, None]

    num_points, num_vars = pred.shape
    if num_points > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(num_points, size=max_points, replace=False)
        pred, true = pred[idx], true[idx]

    fig, axes = plt.subplots(1, num_vars, figsize=(6 * num_vars, 5.5), squeeze=False)
    axes = axes[0]

    for i, ax in enumerate(axes):
        p, t = pred[:, i], true[:, i]
        name = var_names[i] if i < len(var_names) else f"var_{i}"

        ax.scatter(t, p, s=2, alpha=0.25, color="tab:blue", linewidths=0)

        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")

        rel_l2 = np.sqrt(np.mean((p - t) ** 2)) / (np.sqrt(np.mean(t**2)) + 1e-12)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{title_prefix}{name} (rel. L2 = {rel_l2:.4f})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfg, checkpoint_dir = resolve_model_and_checkpoint(cfg)
    assert_surface_only(model_cfg)

    DistributedManager.initialize()
    dm = DistributedManager()

    checkpoint_files = list(Path(checkpoint_dir).glob("*.pt")) if Path(checkpoint_dir).exists() else []
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_dir}'. Train a model first (train.py), "
            "or set eval.model_config / eval.model_checkpoint_dir to point at an existing run."
        )
    print(f"Loading checkpoint from: {checkpoint_dir}")

    num_surf_vars, num_global_features = get_num_vars(model_cfg)
    surf_factors = load_scaling_factors(model_cfg)

    model = DoMINO(
        input_features=3,
        output_features_vol=None,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=model_cfg.model,
    ).to(dm.device)
    load_checkpoint(checkpoint_dir, models=model, device=dm.device)
    model.eval()

    keys_to_read, keys_to_read_if_available = get_keys_to_read(model_cfg, get_ground_truth=True)

    # eval.* (where the test data / outputs live) always comes from the live
    # cfg this script was invoked with; everything else about the model
    # (model_cfg.model / .variables / .data) comes from the resolved model
    # config -- see module docstring.
    dataset_cfg = model_cfg
    dataset_cfg.eval = cfg.eval
    dataset_cfg.data_processor = cfg.data_processor

    eval_sampling = cfg.eval.get("sampling", False)
    if eval_sampling:
        # create_domino_dataset reads the sample count from cfg.model, not
        # from a dedicated eval.* value, so borrow the field for this call.
        # See conf/config.yaml eval.sampling for why you'd want this: it's a
        # workaround for evaluating without RAPIDS cuML available.
        dataset_cfg.model.surface_points_sample = cfg.eval.surface_points_sample
        print(
            f"eval.sampling=true: evaluating a random {cfg.eval.surface_points_sample}-point "
            "subset of each case's surface mesh, not the full mesh."
        )

    test_dataloader = create_domino_dataset(
        dataset_cfg,
        phase="test",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=None,
        surf_factors=surf_factors,
        normalize_coordinates=model_cfg.data.normalize_coordinates,
        sample_in_bbox=model_cfg.data.sample_in_bbox,
        sampling=eval_sampling,
    )

    case_names = [p.stem for p in getattr(test_dataloader.dataset, "_filenames", [])]
    if len(case_names) != len(test_dataloader):
        case_names = [f"case_{i:04d}" for i in range(len(test_dataloader))]

    save_path = to_absolute_path(cfg.eval.save_path)
    create_directory(save_path)

    surface_variable_names = list(model_cfg.variables.surface.solution.keys())

    # See conf/config.yaml's data.temperature_reference_key/_fit docstring: if
    # either is set, surface_fields in this data is a delta from a
    # boundary-condition-derived reference (scripts/rebase_surface_temperature.py),
    # not absolute temperature. Metrics below are computed on the delta
    # values as-is (that's what the model actually predicts); reference_fn,
    # if set, is only used further down to add the reference back for
    # human-facing .vtp/plot output.
    bc_names = list(model_cfg.variables.global_parameters.keys())
    reference_fit_path = model_cfg.data.get("temperature_reference_fit", None)
    temperature_reference_key = model_cfg.data.get("temperature_reference_key", None)
    reference_fn = None
    if reference_fit_path:
        with open(to_absolute_path(reference_fit_path)) as f:
            reference_fit = json.load(f)
        if reference_fit["bc_names"] != bc_names:
            raise ValueError(
                f"{reference_fit_path} was fit against BC order {reference_fit['bc_names']}, "
                f"but this run's config declares {bc_names} -- these must match."
            )
        fit_coeffs = np.array([reference_fit["coefficients"][name] for name in bc_names])
        fit_intercept = reference_fit["intercept"]
        reference_fn = lambda bc_values: float(np.dot(fit_coeffs, bc_values) + fit_intercept)
        print(
            f"data.temperature_reference_fit={reference_fit_path!r}: adding the fitted "
            "linear-combination reference back to predictions/targets before saving "
            ".vtp/plot output (metrics above are still computed on the delta values the "
            "model was trained on)."
        )
    elif temperature_reference_key:
        reference_bc_index = bc_names.index(temperature_reference_key)
        reference_fn = lambda bc_values: float(bc_values[reference_bc_index])
        print(
            f"data.temperature_reference_key={temperature_reference_key!r}: adding it back "
            "to predictions/targets before saving .vtp/plot output (metrics above are still "
            "computed on the delta values the model was trained on)."
        )

    l2_all = []
    l2_centered_all = []
    all_pred, all_true = [], []
    for idx, case_name in enumerate(case_names):
        with torch.no_grad():
            batch = test_dataloader[idx]
            _, prediction_surf = model(batch)

        metrics = compute_l2(prediction_surf, batch, test_dataloader)
        rel_l2 = metrics["l2_surf_temperature"].item()
        rel_l2_centered = metrics["l2_surf_temperature_centered"].item()
        l2_all.append(rel_l2)
        l2_centered_all.append(rel_l2_centered)
        print(
            f"[{case_name}] relative L2 (surface {surface_variable_names}): {rel_l2:.5f}  "
            f"(mean-centered, i.e. spatial-pattern-only: {rel_l2_centered:.5f})"
        )

        _, pred_phys = test_dataloader.unscale_model_outputs(surface_fields=prediction_surf)
        _, true_phys = test_dataloader.unscale_model_outputs(surface_fields=batch["surface_fields"])

        if "surface_min_max" in batch:
            coords_phys = unnormalize(
                batch["surface_mesh_centers"],
                batch["surface_min_max"][:, 1],
                batch["surface_min_max"][:, 0],
            )
        else:
            coords_phys = batch["surface_mesh_centers"]

        pred_np = pred_phys[0].cpu().numpy()
        true_np = true_phys[0].cpu().numpy()
        coords_np = coords_phys[0].cpu().numpy()

        if reference_fn is not None:
            case_bc_values = batch["global_params_values"][0, :, 0].cpu().numpy()
            reference_value = reference_fn(case_bc_values)
            pred_np = pred_np + reference_value
            true_np = true_np + reference_value

        cloud = pv.PolyData(coords_np)
        cloud[f"{surface_variable_names[0]}Pred"] = pred_np
        cloud[f"{surface_variable_names[0]}True"] = true_np
        cloud.save(os.path.join(save_path, f"{case_name}_predicted.vtp"))

        if cfg.eval.save_plots:
            plot_pred_vs_true(
                pred_np,
                true_np,
                surface_variable_names,
                os.path.join(save_path, f"{case_name}_pred_vs_true.png"),
                title_prefix=f"{case_name} - ",
            )
            all_pred.append(pred_np)
            all_true.append(true_np)

    if cfg.eval.save_plots and all_pred:
        combined_path = os.path.join(save_path, f"all_cases_pred_vs_true_rank{dm.rank}.png")
        plot_pred_vs_true(
            np.concatenate(all_pred, axis=0),
            np.concatenate(all_true, axis=0),
            surface_variable_names,
            combined_path,
            title_prefix="All cases - ",
        )
        print(f"Saved combined pred-vs-actual plot to '{combined_path}'")

    print(f"Mean relative L2 over {len(l2_all)} case(s): {np.mean(l2_all):.5f}")
    print(
        f"Mean mean-centered relative L2 (spatial-pattern-only) over {len(l2_centered_all)} "
        f"case(s): {np.mean(l2_centered_all):.5f}"
    )
    print(
        "If the plain relative L2 above looks good but the mean-centered one is much worse, "
        "the model is mostly reproducing each case's baseline temperature rather than the "
        "spatial pattern across its surface -- see the pred-vs-true plots and README for more."
    )


if __name__ == "__main__":
    main()
