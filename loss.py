"""Loss functions for the surface-temperature DoMINO surrogate.

Surface-only, single scalar field (temperature): see utils.assert_surface_only.
The physics-informed (Navier-Stokes residual) and drag/lift integral losses
from the original DrivAer aerodynamics surrogate do not apply here -- there
is no velocity/pressure/shear solution to differentiate or integrate -- and
have been removed rather than left in as dead, untested code paths.
"""

from typing import Literal

import torch

from utils import nvtx_range


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Mean squared (or root-mean-squared) error between predicted and target temperature."""
    numerator = torch.mean((output - target) ** 2.0)

    if loss_type == "mse":
        return numerator
    elif loss_type == "rmse":
        denom = torch.mean((target - torch.mean(target, (0, 1))) ** 2.0)
        return numerator / denom
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def loss_fn_area(
    output: torch.Tensor,
    target: torch.Tensor,
    area: torch.Tensor,
    area_scaling_factor: float,
    loss_type: Literal["mse", "rmse"],
) -> torch.Tensor:
    """
    Area-weighted variant of loss_fn_surface: weights each surface point's
    error contribution by its mesh element area (scaled by
    `area_scaling_factor`, e.g. model.loss_function.area_weighing_factor).
    Reported for monitoring; see compute_loss_dict for how it factors into
    the training loss.
    """
    weight = area * area_scaling_factor

    output_scaled = output * weight
    target_scaled = target * weight

    numerator = torch.mean((output_scaled - target_scaled) ** 2.0, dim=(0, 1))

    if loss_type == "rmse":
        denom = torch.mean((target_scaled - torch.mean(target_scaled, (0, 1))) ** 2.0)
        denom = denom.clamp_min(1e-4)
        return numerator / denom
    return numerator


def compute_loss_dict(
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type,
    surf_loss_scaling: float,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the total training loss and a dict of its components (for logging).

    Total loss = surf_loss_scaling * (point-wise MSE/RMSE loss + area-weighted loss).
    The area-weighted term is computed under torch.no_grad() -- it is folded
    into the reported total for a single "how are we doing" number, but
    contributes zero gradient, so training is driven entirely by the
    point-wise loss_surf term.
    """
    with nvtx_range("Loss Calculation"):
        target_surf = batch_inputs["surface_fields"]
        surface_areas = torch.unsqueeze(batch_inputs["surface_areas"], -1)

        loss_surf = loss_fn_surface(prediction_surf, target_surf, loss_fn_type.loss_type)
        loss_surf = loss_surf * surf_loss_scaling

        with torch.no_grad():
            loss_surf_area = loss_fn_area(
                prediction_surf,
                target_surf,
                surface_areas,
                area_scaling_factor=loss_fn_type.area_weighing_factor,
                loss_type=loss_fn_type.loss_type,
            )
            if loss_fn_type.loss_type == "mse":
                loss_surf_area = loss_surf_area * surf_loss_scaling

        total_loss = loss_surf + loss_surf_area

        loss_dict = {
            "loss_surf": loss_surf,
            "loss_surf_area": loss_surf_area,
            "total_loss": total_loss,
        }

    return total_loss, loss_dict
