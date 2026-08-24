"""Shared helpers for the surface-temperature DoMINO surrogate.

This project only supports ``cfg.model.model_type == "surface"``: the .zarr
dataset (see ref_material/zarr_structure.md) has a single scalar
"surface_fields" array (temperature) and no volume solution. Functions below
assert that up front rather than silently accepting "volume"/"combined" and
producing a half-working pipeline.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager

# torch.cuda.nvtx.range() raises RuntimeError on a CPU-only PyTorch build
# ("NVTX functions not installed"), so profiling markers are only emitted
# when a CUDA device is actually available.
_NVTX_ENABLED = torch.cuda.is_available()


@contextmanager
def nvtx_range(msg: str):
    """CPU-safe wrapper around torch.cuda.nvtx.range for profiling markers."""
    if _NVTX_ENABLED:
        with torch.cuda.nvtx.range(msg):
            yield
    else:
        yield


def assert_surface_only(cfg: DictConfig) -> None:
    if cfg.model.model_type != "surface":
        raise ValueError(
            "This project only implements the surface-only DoMINO pipeline "
            f"(no volume solution in the dataset), got model.model_type="
            f"{cfg.model.model_type!r}. Set model.model_type=surface."
        )


@dataclass
class ScalingFactors:
    """
    Mean/std/min/max statistics computed over the training dataset, used to
    normalize model targets and denormalize model outputs.

    Attributes:
        mean: Dictionary mapping keys to mean numpy arrays.
        std: Dictionary mapping keys to standard deviation numpy arrays.
        min_val: Dictionary mapping keys to minimum value numpy arrays.
        max_val: Dictionary mapping keys to maximum value numpy arrays.
        field_keys: List of field keys for which statistics were computed.
    """

    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    min_val: Dict[str, np.ndarray]
    max_val: Dict[str, np.ndarray]
    field_keys: list[str]

    def save(self, filepath: str | Path) -> None:
        """Save scaling factors to a pickle file."""
        import pickle

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "ScalingFactors":
        """Load scaling factors from a pickle file."""
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def summary(self) -> str:
        """Generate a human-readable summary of the scaling factors."""
        lines = ["Scaling Factors Summary:", f"Field Keys: {self.field_keys}"]
        for key in self.field_keys:
            lines.append(f"\n{key}:")
            lines.append(f"  Shape: {self.mean[key].shape}")
            lines.append(f"  Mean: {self.mean[key]}")
            lines.append(f"  Std: {self.std[key]}")
            lines.append(f"  Min: {self.min_val[key]}")
            lines.append(f"  Max: {self.max_val[key]}")
        return "\n".join(lines)


def get_keys_to_read(cfg: DictConfig, get_ground_truth: bool = True):
    """
    Determine which arrays to read from each case's .zarr group.

    `global_params_reference` is read from each case's own .zarr file (its
    order there must match `global_params_values`' order, whatever that is
    -- both are written by the same data-curation pipeline, so this is the
    authoritative source). The `keys_to_read_if_available` fallback below is
    only used if a specific case's .zarr is missing that array; built from
    cfg.variables.global_parameters in *declared* order, matching this
    project's documented convention (see conf/config.yaml). Note this means
    the fallback is only correct if that convention actually holds for the
    case at hand -- unlike the primary, per-case path, there's no way to
    verify it here.
    """
    assert_surface_only(cfg)

    keys_to_read = [
        "stl_coordinates",
        "stl_centers",
        "stl_faces",
        "stl_areas",
        "global_params_values",
        "global_params_reference",
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
    ]
    if get_ground_truth:
        keys_to_read.append("surface_fields")

    cfg_params_vec = []
    for key in cfg.variables.global_parameters:
        param = cfg.variables.global_parameters[key]
        if param.type == "vector":
            cfg_params_vec.extend(param.reference)
        else:
            cfg_params_vec.append(param.reference)

    keys_to_read_if_available = {
        "global_params_reference": torch.tensor(cfg_params_vec).reshape(-1, 1),
    }

    return keys_to_read, keys_to_read_if_available


def get_num_vars(cfg: DictConfig) -> tuple[int, int]:
    """
    Count the surface and global-parameter feature dimensions implied by the
    config (vector variables contribute 3 components, scalars contribute 1).

    Returns:
        (num_surf_vars, num_global_features)
    """
    assert_surface_only(cfg)

    num_surf_vars = 0
    for name in cfg.variables.surface.solution:
        num_surf_vars += 3 if cfg.variables.surface.solution[name] == "vector" else 1

    num_global_features = 0
    for name in cfg.variables.global_parameters:
        param = cfg.variables.global_parameters[name]
        if param.type == "vector":
            num_global_features += len(param.reference)
        elif param.type == "scalar":
            num_global_features += 1
        else:
            raise ValueError(f"Unknown global parameter type: {param.type!r}")

    return num_surf_vars, num_global_features


def load_scaling_factors(cfg: DictConfig, logger=None) -> torch.Tensor:
    """Load the surface-field scaling factors computed by compute_statistics.py."""
    pickle_path = os.path.join(cfg.data.scaling_factors)

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        if logger is not None:
            logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaling factors not found at: {pickle_path}; run compute_statistics.py first."
        )

    if cfg.model.normalization == "min_max_scaling":
        surf_factors = np.asarray(
            [
                scaling_factors.max_val["surface_fields"],
                scaling_factors.min_val["surface_fields"],
            ]
        )
    elif cfg.model.normalization == "mean_std_scaling":
        surf_factors = np.asarray(
            [
                scaling_factors.mean["surface_fields"],
                scaling_factors.std["surface_fields"],
            ]
        )
    else:
        raise ValueError(f"Invalid normalization mode: {cfg.model.normalization}")

    dm = DistributedManager()
    return torch.from_numpy(surf_factors).to(dm.device, dtype=torch.float32)


def compute_l2(
    pred_surface: Optional[torch.Tensor],
    batch: dict,
    dataloader,
) -> dict[str, torch.Tensor]:
    """
    Compute the relative L2 norm between prediction and target surface
    temperature, unscaling both back to physical units first.
    """
    _, target_surface = dataloader.unscale_model_outputs(surface_fields=batch["surface_fields"])
    _, pred_surface = dataloader.unscale_model_outputs(surface_fields=pred_surface)
    return metrics_fn_surface(pred_surface, target_surface)


def metrics_fn_surface(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Relative L2 error of predicted vs. target surface temperature."""
    l2_num = torch.sqrt(torch.sum((pred - target) ** 2, dim=1))
    l2_denom = torch.sqrt(torch.sum(target**2, dim=1))
    l2 = l2_num / l2_denom
    return {"l2_surf_temperature": torch.mean(l2[:, 0])}


def all_reduce_dict(
    metrics: dict[str, torch.Tensor], dm: DistributedManager
) -> dict[str, torch.Tensor]:
    """Average a dictionary of metrics across all distributed processes."""
    if dm.world_size == 1:
        return metrics

    for key, value in metrics.items():
        dist.all_reduce(value)
        metrics[key] = value / dm.world_size

    return metrics
