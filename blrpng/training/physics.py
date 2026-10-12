from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def laplacian(field: torch.Tensor) -> torch.Tensor:
    height, width = field.shape[-2:]
    hx = 1.0 / (height - 1)
    hy = 1.0 / (width - 1)
    center = field[..., 1:-1, 1:-1]
    return (
        (field[..., 2:, 1:-1] - 2 * center + field[..., :-2, 1:-1]) / hx**2
        + (field[..., 1:-1, 2:] - 2 * center + field[..., 1:-1, :-2]) / hy**2
    )


def allen_cahn_defect(previous: torch.Tensor, current: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    center = current[..., 1:-1, 1:-1]
    return (
        (center - previous[..., 1:-1, 1:-1]) / float(params["dt"])
        - float(params.get("eps", params.get("epsilon"))) ** 2 * laplacian(current)
        + float(params["gamma"]) * (center**3 - center)
    )


def poisson_defect(previous: torch.Tensor, current: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    conditions = params.get("condition_fields")
    forcing = conditions[:, 0:1] if conditions is not None else params.get("forcing")
    if forcing is None:
        return -laplacian(current)
    return -laplacian(current) - forcing[..., 1:-1, 1:-1]


def gray_scott_defect(previous: torch.Tensor, current: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    u, v = current[:, 0:1], current[:, 1:2]
    pu, pv = previous[:, 0:1], previous[:, 1:2]
    ui, vi = u[..., 1:-1, 1:-1], v[..., 1:-1, 1:-1]
    conditions = params.get("condition_fields")
    if conditions is None:
        feed, kill = float(params["feed"]), float(params["kill"])
    else:
        feed = conditions[:, 0:1, 1:-1, 1:-1]
        kill = conditions[:, 1:2, 1:-1, 1:-1]
    ru = (
        (ui - pu[..., 1:-1, 1:-1]) / float(params["dt"])
        - float(params["diffusion_u"]) * laplacian(u)
        + ui * vi.square() - feed * (1.0 - ui)
    )
    rv = (
        (vi - pv[..., 1:-1, 1:-1]) / float(params["dt"])
        - float(params["diffusion_v"]) * laplacian(v)
        - ui * vi.square() + (feed + kill) * vi
    )
    return torch.cat((ru, rv), dim=1)


def hyperelastic_defect(previous: torch.Tensor, current: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """Interior equilibrium residual for a compressible Neo-Hookean solid."""
    if current.shape[1] != 2:
        raise ValueError("Hyperelastic defect expects [ux,uy].")
    h = 1.0 / (current.shape[-1] - 1)
    ux, uy = current[:, 0:1], current[:, 1:2]
    dux_dx = (ux[..., 2:, 1:-1] - ux[..., :-2, 1:-1]) / (2.0 * h)
    dux_dy = (ux[..., 1:-1, 2:] - ux[..., 1:-1, :-2]) / (2.0 * h)
    duy_dx = (uy[..., 2:, 1:-1] - uy[..., :-2, 1:-1]) / (2.0 * h)
    duy_dy = (uy[..., 1:-1, 2:] - uy[..., 1:-1, :-2]) / (2.0 * h)
    f11, f12 = 1.0 + dux_dx, dux_dy
    f21, f22 = duy_dx, 1.0 + duy_dy
    determinant = (f11 * f22 - f12 * f21).clamp_min(1e-6)
    inverse_t11, inverse_t12 = f22 / determinant, -f21 / determinant
    inverse_t21, inverse_t22 = -f12 / determinant, f11 / determinant
    conditions = params.get("condition_fields")
    if conditions is None:
        mu: float | torch.Tensor = float(params.get("mu", 1.0))
        lam: float | torch.Tensor = float(params.get("lambda", 2.0))
    else:
        mu = conditions[:, 0:1, 1:-1, 1:-1]
        lam = conditions[:, 1:2, 1:-1, 1:-1]
    log_j = torch.log(determinant)
    p11 = mu * (f11 - inverse_t11) + lam * log_j * inverse_t11
    p12 = mu * (f12 - inverse_t12) + lam * log_j * inverse_t12
    p21 = mu * (f21 - inverse_t21) + lam * log_j * inverse_t21
    p22 = mu * (f22 - inverse_t22) + lam * log_j * inverse_t22
    div_x = (
        (p11[..., 2:, 1:-1] - p11[..., :-2, 1:-1])
        + (p12[..., 1:-1, 2:] - p12[..., 1:-1, :-2])
    ) / (2.0 * h)
    div_y = (
        (p21[..., 2:, 1:-1] - p21[..., :-2, 1:-1])
        + (p22[..., 1:-1, 2:] - p22[..., 1:-1, :-2])
    ) / (2.0 * h)
    return torch.cat((div_x, div_y), dim=1)


def navier_stokes_defect(previous: torch.Tensor, current: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """Collocated diagnostic residual used for differentiable fine-grid calibration."""
    u, v, p = current[:, 0:1], current[:, 1:2], current[:, 2:3]
    pu, pv = previous[:, 0:1], previous[:, 1:2]
    h = 1.0 / (current.shape[-1] - 1)
    ui, vi = u[..., 1:-1, 1:-1], v[..., 1:-1, 1:-1]
    dudx = (u[..., 2:, 1:-1] - u[..., :-2, 1:-1]) / (2 * h)
    dudy = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2 * h)
    dvdx = (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2 * h)
    dvdy = (v[..., 1:-1, 2:] - v[..., 1:-1, :-2]) / (2 * h)
    dpdx = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / (2 * h)
    dpdy = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / (2 * h)
    conditions = params.get("condition_fields")
    viscosity: float | torch.Tensor
    time_step: float | torch.Tensor
    force_u: float | torch.Tensor = 0.0
    force_v: float | torch.Tensor = 0.0
    if conditions is None:
        viscosity = float(params.get("viscosity", 1.0 / float(params.get("reynolds", 100.0))))
        time_step = float(params["dt"])
    else:
        if conditions.shape[1] < 2:
            raise ValueError(
                "Navier--Stokes condition_fields require at least [viscosity, dt]."
            )
        viscosity = conditions[:, 0:1, 1:-1, 1:-1]
        time_step = conditions[:, 1:2, 1:-1, 1:-1].clamp_min(1e-12)
        if conditions.shape[1] >= 4:
            force_u = conditions[:, 2:3, 1:-1, 1:-1]
            force_v = conditions[:, 3:4, 1:-1, 1:-1]
    ru = (
        (ui - pu[..., 1:-1, 1:-1]) / time_step
        + ui * dudx
        + vi * dudy
        + dpdx
        - viscosity * laplacian(u)
        - force_u
    )
    rv = (
        (vi - pv[..., 1:-1, 1:-1]) / time_step
        + ui * dvdx
        + vi * dvdy
        + dpdy
        - viscosity * laplacian(v)
        - force_v
    )
    divergence = dudx + dvdy
    return torch.cat((ru, rv, divergence), dim=1)


def relative_defect_loss(
    equation: str,
    previous: torch.Tensor,
    current: torch.Tensor,
    params: dict[str, Any],
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    functions = {
        "poisson": poisson_defect,
        "allen_cahn": allen_cahn_defect,
        "gray_scott": gray_scott_defect,
        "hyperelastic": hyperelastic_defect,
        "navier_stokes": navier_stokes_defect,
    }
    if equation not in functions:
        raise ValueError(f"Unsupported defect equation: {equation}")
    defect = functions[equation](previous, current, params)
    scale = current[..., 1:-1, 1:-1].flatten(1).norm(dim=1).clamp_min(epsilon)
    return (defect.flatten(1).norm(dim=1) / scale).mean()


def restriction_commutator_loss(
    fine_prediction: torch.Tensor,
    coarse_prediction: torch.Tensor,
    *,
    field_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    restricted = F.interpolate(
        fine_prediction,
        size=coarse_prediction.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    if field_scales is not None:
        scales = field_scales.to(restricted).view(1, -1, 1, 1)
        restricted, coarse_prediction = restricted / scales, coarse_prediction / scales
    numerator = (restricted - coarse_prediction).flatten(1).norm(dim=1)
    denominator = restricted.flatten(1).norm(dim=1).clamp_min(1e-8)
    return (numerator / denominator).mean()
