"""Compute and save scaling factors for the surface-temperature DoMINO dataset.

Computes mean/std/min/max for the surface temperature field (used to
normalize training targets in train.py, per model.normalization) plus a
couple of coordinate arrays (useful as a sanity check against
data.bounding_box_surface, though they aren't used directly for
normalization -- that's controlled by the bounding box in config.yaml).
Run this once before train.py; it writes to data.scaling_factors.
"""

import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes.cae.domino_datapipe import compute_scaling_factors
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from utils import ScalingFactors, assert_surface_only

TARGET_KEYS = ["surface_fields", "stl_centers", "surface_mesh_centers"]


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compute and save scaling factors, or report that they already exist."""
    assert_surface_only(cfg)

    DistributedManager.initialize()
    dm = DistributedManager()

    logger = PythonLogger("ComputeStatistics")
    logger = RankZeroLoggingWrapper(logger, dm)
    logger.info("Starting scaling factors computation")
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    pickle_path = cfg.data.scaling_factors
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    if dm.world_size > 1:
        torch.distributed.barrier()

    try:
        ScalingFactors.load(pickle_path)
        logger.info(f"Scaling factors already exist at: {pickle_path}; skipping.")
        return
    except FileNotFoundError:
        pass

    logger.info("Computing scaling factors from dataset...")
    start_time = time.perf_counter()

    mean, std, min_val, max_val = compute_scaling_factors(
        cfg=cfg,
        input_path=cfg.data.input_dir,
        target_keys=TARGET_KEYS,
        max_samples=cfg.data.max_samples_for_statistics,
    )
    scaling_factors = ScalingFactors(
        mean={k: v.cpu().numpy() for k, v in mean.items()},
        std={k: v.cpu().numpy() for k, v in std.items()},
        min_val={k: v.cpu().numpy() for k, v in min_val.items()},
        max_val={k: v.cpu().numpy() for k, v in max_val.items()},
        field_keys=TARGET_KEYS,
    )

    compute_time = time.perf_counter() - start_time
    logger.info(f"Scaling factors computation completed in {compute_time:.2f} seconds")

    if dm.rank == 0:
        scaling_factors.save(pickle_path)
        logger.info(f"Scaling factors saved to: {pickle_path}")

        summary_path = os.path.join(os.path.dirname(pickle_path), "scaling_factors_summary.txt")
        with open(summary_path, "w") as f:
            f.write(scaling_factors.summary())
        logger.info(f"Summary report saved to: {summary_path}")

    logger.info("Scaling factors computation completed successfully!")


if __name__ == "__main__":
    main()
