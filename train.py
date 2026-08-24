"""Train the surface-temperature DoMINO surrogate.

Surface-only: reads .zarr cases (see ref_material/zarr_structure.md) with a
single scalar "surface_fields" (temperature) target and 6 boundary-condition
values in "global_params_values". Run `compute_statistics.py` once first to
generate the scaling factors this script loads.
"""

import math
import os
import time

import hydra
import mlflow
import torch
import torchinfo
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import create_directory

import cuml_knn_patch  # noqa: F401  -- see that file: works around a RAPIDS/physicsnemo cuML API mismatch
from loss import compute_loss_dict
from utils import (
    all_reduce_dict,
    assert_surface_only,
    compute_l2,
    get_keys_to_read,
    get_num_vars,
    load_scaling_factors,
    nvtx_range,
)

try:
    from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex, nvmlInit

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


def _gpu_memory_gb(gpu_handle) -> float | None:
    """Current GPU memory usage in GB, or None if no GPU/NVML is available."""
    if gpu_handle is None:
        return None
    return nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024**3)


def validation_step(
    dataloader,
    model,
    device,
    logger,
    tb_writer,
    epoch_index,
    loss_fn_type,
    surf_loss_scaling,
    autocast_enabled,
):
    """Run one validation epoch and return the average loss."""
    dm = DistributedManager()
    running_vloss = 0.0
    metrics = None

    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            with autocast(device.type, enabled=autocast_enabled, cache_enabled=False):
                _, prediction_surf = model(batch)
                loss, _ = compute_loss_dict(
                    prediction_surf, batch, loss_fn_type, surf_loss_scaling
                )

            running_vloss += loss.item()
            local_metrics = compute_l2(prediction_surf, batch, dataloader)
            if metrics is None:
                metrics = local_metrics
            else:
                metrics = {k: metrics[k] + local_metrics[k] for k in metrics}

    avg_vloss = running_vloss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
    metrics = all_reduce_dict(metrics, dm)

    if dm.rank == 0:
        logger.info(f"Device {device}, batch: {i_batch + 1}, VAL loss: {avg_vloss:.5f}")
        for key, value in metrics.items():
            tb_writer.add_scalar(f"L2 Metrics/val/{key}", value, epoch_index)
            mlflow.log_metric(f"L2 Metrics/val/{key}", value, step=epoch_index)
        mlflow.log_metric("Loss/val", avg_vloss, step=epoch_index)

    return avg_vloss


def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    loss_fn_type,
    surf_loss_scaling,
    autocast_enabled,
    grad_clip_enabled,
    grad_max_norm,
):
    """Run one training epoch and return the average loss."""
    dm = DistributedManager()

    running_loss = 0.0
    metrics = None
    start_time = time.perf_counter()
    io_start_time = time.perf_counter()

    for i_batch, batch in enumerate(dataloader):
        io_end_time = time.perf_counter()

        with autocast(device.type, enabled=autocast_enabled, cache_enabled=False):
            with nvtx_range("Model Forward Pass"):
                _, prediction_surf = model(batch)

            loss, loss_dict = compute_loss_dict(
                prediction_surf, batch, loss_fn_type, surf_loss_scaling
            )

            local_metrics = compute_l2(prediction_surf, batch, dataloader)
            if metrics is None:
                metrics = local_metrics
            else:
                metrics = {k: metrics[k] + local_metrics[k] for k in metrics}

        scaler.scale(loss).backward()

        if grad_clip_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.detach().item()
        elapsed_time = time.perf_counter() - start_time
        io_time = io_end_time - io_start_time
        start_time = time.perf_counter()

        gpu_mem_used = _gpu_memory_gb(gpu_handle)
        mem_str = f"{gpu_mem_used:.3f} Gb" if gpu_mem_used is not None else "n/a"

        loss_str = "  " + "\t".join(f"{k.replace('loss_', ''):<10}" for k in loss_dict) + "\n"
        loss_str += (
            "  " + "\t".join(f"{v.detach().item():<10.3e}" for v in loss_dict.values()) + "\n"
        )
        logger.info(
            f"Device {device}, batch processed: {i_batch + 1}\n"
            f"{loss_str}"
            f"  GPU memory used: {mem_str}\n"
            f"  Timings: (IO: {io_time:.2f}, Model: {elapsed_time - io_time:.2f}, "
            f"Total: {elapsed_time:.2f})s\n"
        )
        io_start_time = time.perf_counter()

    last_loss = running_loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
    metrics = all_reduce_dict(metrics, dm)

    if dm.rank == 0:
        logger.info(f"Device {device}, batch: {i_batch + 1}, loss: {last_loss:.5f}")
        tb_step = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_step)
        for key, value in metrics.items():
            tb_writer.add_scalar(f"L2 Metrics/train/{key}", value, epoch_index)
            mlflow.log_metric(f"L2 Metrics/train/{key}", value, step=epoch_index)
        mlflow.log_metric("Loss/train", last_loss, step=epoch_index)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for DoMINO surface-temperature training."""
    assert_surface_only(cfg)

    DistributedManager.initialize()
    dm = DistributedManager()

    mlflow.set_tracking_uri(f"sqlite:///{to_absolute_path(cfg.project.mlflow_db)}")
    if dm.rank == 0:
        mlflow.set_experiment(cfg.project.name)
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_name=f"{cfg.project.name}_tag{cfg.exp_tag}")

    try:
        if dm.rank == 0:
            flat_cfg = {}

            def _flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}{k}"
                    if isinstance(v, dict) or hasattr(v, "items"):
                        _flatten(v, prefix=f"{key}.")
                    else:
                        flat_cfg[key] = v

            _flatten(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)

            # sort_keys=False: test.py's eval.model_config loads this file back
            # via OmegaConf.load() to evaluate this checkpoint later, and dict
            # iteration order matters (see utils.get_keys_to_read) -- alphabetic
            # sorting here would silently reorder cfg.variables.global_parameters
            # relative to how it was declared when this run actually trained.
            config_path = os.path.join(cfg.output, "resolved_config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg, sort_keys=False))
            mlflow.log_artifact(config_path)

        gpu_handle = None
        if _NVML_AVAILABLE and dm.device.type == "cuda":
            nvmlInit()
            gpu_handle = nvmlDeviceGetHandleByIndex(dm.device.index)

        logger = PythonLogger("Train")
        logger = RankZeroLoggingWrapper(logger, dm)
        logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

        ######################################################
        # Scaling factors and model sizing
        ######################################################
        surf_factors = load_scaling_factors(cfg, logger)
        num_surf_vars, num_global_features = get_num_vars(cfg)
        keys_to_read, keys_to_read_if_available = get_keys_to_read(cfg, get_ground_truth=True)

        ######################################################
        # Datasets
        ######################################################
        train_dataloader = create_domino_dataset(
            cfg,
            phase="train",
            keys_to_read=keys_to_read,
            keys_to_read_if_available=keys_to_read_if_available,
            vol_factors=None,
            surf_factors=surf_factors,
            normalize_coordinates=cfg.data.normalize_coordinates,
            sample_in_bbox=cfg.data.sample_in_bbox,
            sampling=cfg.data.sampling,
        )
        train_sampler = DistributedSampler(
            train_dataloader,
            num_replicas=dm.world_size,
            rank=dm.rank,
            **cfg.train.sampler,
        )

        val_dataloader = create_domino_dataset(
            cfg,
            phase="val",
            keys_to_read=keys_to_read,
            keys_to_read_if_available=keys_to_read_if_available,
            vol_factors=None,
            surf_factors=surf_factors,
            normalize_coordinates=cfg.data.normalize_coordinates,
            sample_in_bbox=cfg.data.sample_in_bbox,
            sampling=cfg.data.sampling,
        )
        val_sampler = DistributedSampler(
            val_dataloader,
            num_replicas=dm.world_size,
            rank=dm.rank,
            **cfg.val.sampler,
        )

        ######################################################
        # Model
        ######################################################
        model = DoMINO(
            input_features=3,
            output_features_vol=None,
            output_features_surf=num_surf_vars,
            global_features=num_global_features,
            model_parameters=cfg.model,
        ).to(dm.device)

        logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

        if dm.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[dm.local_rank] if dm.device.type == "cuda" else None,
                output_device=dm.device if dm.device.type == "cuda" else None,
                broadcast_buffers=dm.broadcast_buffers,
                find_unused_parameters=dm.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        ######################################################
        # Optimizer, scheduler, AMP
        ######################################################
        optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}.get(
            cfg.train.optimizer.name
        )
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer.name}")
        optimizer = optimizer_class(
            model.parameters(),
            lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay,
        )

        if cfg.train.lr_scheduler.name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg.train.lr_scheduler.milestones,
                gamma=cfg.train.lr_scheduler.gamma,
            )
        elif cfg.train.lr_scheduler.name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.lr_scheduler.T_max,
                eta_min=cfg.train.lr_scheduler.eta_min,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {cfg.train.lr_scheduler.name}")

        # torch.amp.GradScaler/autocast auto-disable themselves when the target
        # device isn't CUDA, so this is safe to construct on a CPU-only machine.
        scaler = GradScaler(device=dm.device.type, enabled=cfg.train.amp.enabled)

        ######################################################
        # Output directories
        ######################################################
        writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

        model_save_path = os.path.join(cfg.output, "models")
        best_model_path = os.path.join(model_save_path, "best_model")

        if dm.rank == 0:
            create_directory(model_save_path)
            create_directory(best_model_path)
        if dm.world_size > 1:
            torch.distributed.barrier()

        ######################################################
        # Resume from checkpoint, if available
        ######################################################
        init_epoch = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dm.device,
        )
        if init_epoch != 0:
            init_epoch += 1  # Start with the next epoch

        # physicsnemo checkpoint filenames encode {rank}.{epoch}, not the
        # validation loss, so there is no way to recover the previous best
        # val loss from filenames on disk. It simply resets on resume; worst
        # case this re-saves an already-current best_model checkpoint once.
        best_vloss = 1_000_000.0

        ######################################################
        # Training loop
        ######################################################
        epoch_number = init_epoch
        for epoch in range(init_epoch, cfg.train.epochs):
            logger.info(f"Device {dm.device}, epoch {epoch_number}:")

            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            train_dataloader.dataset.set_indices(list(train_sampler))
            val_dataloader.dataset.set_indices(list(val_sampler))

            model.train(True)
            epoch_start_time = time.perf_counter()
            avg_loss = train_epoch(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                tb_writer=writer,
                logger=logger,
                gpu_handle=gpu_handle,
                epoch_index=epoch,
                device=dm.device,
                loss_fn_type=cfg.model.loss_function,
                surf_loss_scaling=cfg.model.surf_loss_scaling,
                autocast_enabled=cfg.train.amp.enabled,
                grad_clip_enabled=cfg.train.amp.clip_grad,
                grad_max_norm=cfg.train.amp.grad_max_norm,
            )
            logger.info(
                f"Device {dm.device}, Epoch {epoch_number} took "
                f"{time.perf_counter() - epoch_start_time:.3f} seconds"
            )

            model.eval()
            avg_vloss = validation_step(
                dataloader=val_dataloader,
                model=model,
                device=dm.device,
                logger=logger,
                tb_writer=writer,
                epoch_index=epoch,
                loss_fn_type=cfg.model.loss_function,
                surf_loss_scaling=cfg.model.surf_loss_scaling,
                autocast_enabled=cfg.train.amp.enabled,
            )
            scheduler.step()

            logger.info(
                f"Device {dm.device} LOSS train {avg_loss:.5f} valid {avg_vloss:.5f} "
                f"Current lr {scheduler.get_last_lr()[0]}"
            )

            if dm.rank == 0:
                writer.add_scalars(
                    "Training vs. Validation Loss",
                    {"Training": avg_loss, "Validation": avg_vloss},
                    epoch_number,
                )
                writer.flush()
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch_number)

            if dm.world_size > 1:
                torch.distributed.barrier()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                if dm.rank == 0:
                    save_checkpoint(
                        to_absolute_path(best_model_path),
                        models=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metadata={"val_loss": best_vloss},
                    )

            if dm.rank == 0:
                logger.info(f"Device {dm.device}, Best val loss {best_vloss}")

            if dm.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0:
                save_checkpoint(
                    to_absolute_path(model_save_path),
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                )

            epoch_number += 1

            if cfg.train.lr_scheduler.name == "CosineAnnealingLR" and math.isclose(
                scheduler.get_last_lr()[0], cfg.train.lr_scheduler.eta_min, rel_tol=1e-6
            ):
                logger.info("LR reached eta_min, ending training")
                break

        return best_vloss

    finally:
        if dm.rank == 0 and mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()
