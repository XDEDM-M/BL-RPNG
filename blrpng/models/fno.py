from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FNOConfig:
    in_channels: int = 4
    out_channels: int = 1
    modes1: int = 24
    modes2: int = 24
    width: int = 96
    layers: int = 6
    probe_mode: str = "per_layer"
    condition_channels: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, modes1, modes2)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self._adapter_kind: str | None = None
        self._adapter_scale = 1.0

    def enable_lora(self, rank: int = 4, alpha: float = 4.0) -> None:
        """Attach a channel-low-rank spectral update without changing base checkpoints."""
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self._adapter_kind is not None:
            raise RuntimeError("A spectral adapter is already enabled.")
        device = self.weight_positive.device
        dtype = self.weight_positive.dtype
        self.lora_a_positive = nn.Parameter(
            torch.zeros(self.in_channels, rank, 1, 1, device=device, dtype=dtype)
        )
        self.lora_b_positive = nn.Parameter(
            torch.zeros(rank, self.out_channels, self.modes1, self.modes2, device=device, dtype=dtype)
        )
        self.lora_a_negative = nn.Parameter(
            torch.zeros(self.in_channels, rank, 1, 1, device=device, dtype=dtype)
        )
        self.lora_b_negative = nn.Parameter(
            torch.zeros(rank, self.out_channels, self.modes1, self.modes2, device=device, dtype=dtype)
        )
        nn.init.normal_(self.lora_a_positive, std=0.02)
        nn.init.normal_(self.lora_a_negative, std=0.02)
        self._adapter_kind = "lora"
        self._adapter_scale = float(alpha) / float(rank)

    def _weight(self, sign: str) -> torch.Tensor:
        base = getattr(self, f"weight_{sign}")
        if self._adapter_kind != "lora":
            return base
        left = getattr(self, f"lora_a_{sign}")
        right = getattr(self, f"lora_b_{sign}")
        delta = torch.einsum("irxy,roxy->ioxy", left, right)
        return base + self._adapter_scale * delta

    @staticmethod
    def _multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        capacity2 = width // 2 + 1
        if self.modes1 > height // 2 or self.modes2 > capacity2:
            raise ValueError(
                f"Configured modes=({self.modes1},{self.modes2}) exceed grid "
                f"{height}x{width} capacities ({height // 2},{capacity2})."
            )
        spectrum = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            capacity2,
            dtype=spectrum.dtype,
            device=x.device,
        )
        output[:, :, : self.modes1, : self.modes2] = self._multiply(
            spectrum[:, :, : self.modes1, : self.modes2],
            self._weight("positive"),
        )
        output[:, :, -self.modes1 :, : self.modes2] = self._multiply(
            spectrum[:, :, -self.modes1 :, : self.modes2],
            self._weight("negative"),
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor, *, activate: bool = True) -> torch.Tensor:
        x = self.norm(self.spectral(x) + self.local(x))
        return F.gelu(x) if activate else x


class FNO2d(nn.Module):
    """Resolution-independent FNO with a probe attached after every block."""

    def __init__(self, config: FNOConfig):
        super().__init__()
        if config.layers < 1:
            raise ValueError("FNO needs at least one Fourier layer.")
        self.config = config
        self.lift = nn.Conv2d(config.in_channels, config.width, 1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock(config.width, config.modes1, config.modes2)
                for _ in range(config.layers)
            ]
        )
        self.probes = nn.ModuleList(
            [nn.Conv2d(config.width, config.out_channels, 1) for _ in range(config.layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(config.width, config.width * 2, 1),
            nn.GELU(),
            nn.Conv2d(config.width * 2, config.out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.lift(x)
        probes: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, activate=index < len(self.blocks) - 1)
            if hasattr(self, "nlt_scales"):
                hidden = hidden * self.nlt_scales[index] + self.nlt_biases[index]
            if hasattr(self, "nlt_mix_left"):
                # Identity-initialized low-rank channel rotation. Unlike a
                # spatial convolution this commutes with grid refinement and
                # adds only O(width*rank) task parameters.
                reduced = torch.einsum(
                    "bchw,rc->brhw", hidden, self.nlt_mix_right[index]
                )
                delta = torch.einsum(
                    "brhw,cr->bchw", reduced, self.nlt_mix_left[index]
                )
                hidden = hidden + self.nlt_mix_scale * delta
            probes[f"layer{index}:hook0"] = self.probes[index](hidden)
        return self.project(hidden), probes

    def path_active_parameters(self, layer: int) -> list[nn.Parameter]:
        """Return trainable representation parameters shared by final/path losses.

        Probe parameters are deliberately excluded: the final prediction does not
        depend on them, so including a probe can make the measured final gradient
        norm zero for parameter-efficient adapters. NLT affine parameters live on
        ``FNO2d`` rather than inside a block and therefore need explicit inclusion.
        """
        if not 0 <= layer < len(self.blocks):
            raise ValueError(f"Path layer {layer} outside [0,{len(self.blocks) - 1}]")
        modules: list[nn.Module] = [self.lift, *self.blocks[: layer + 1]]
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if hasattr(self, "nlt_scales"):
            for collection in (self.nlt_scales, self.nlt_biases):
                for parameter in collection[: layer + 1]:
                    if parameter.requires_grad and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
        if hasattr(self, "nlt_mix_left"):
            for collection in (self.nlt_mix_left, self.nlt_mix_right):
                for parameter in collection[: layer + 1]:
                    if parameter.requires_grad and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
        return parameters

    def enable_nlt(self) -> None:
        """Add trainable neuron-wise affine transforms to a frozen backbone."""
        if hasattr(self, "nlt_scales"):
            raise RuntimeError("NLT adapter is already enabled.")
        device = next(self.parameters()).device
        self.nlt_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.config.width, 1, 1, device=device)) for _ in self.blocks]
        )
        self.nlt_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.config.width, 1, 1, device=device)) for _ in self.blocks]
        )

    def enable_nlt_mixing(self, rank: int, alpha: float) -> None:
        """Augment diagonal NLT with an identity-initialized low-rank basis change."""
        if rank <= 0 or rank > self.config.width:
            raise ValueError(f"NLT mixing rank must be in [1,{self.config.width}].")
        if hasattr(self, "nlt_mix_left"):
            raise RuntimeError("NLT mixing adapter is already enabled.")
        if not hasattr(self, "nlt_scales"):
            self.enable_nlt()
        device = next(self.parameters()).device
        self.nlt_mix_left = nn.ParameterList()
        self.nlt_mix_right = nn.ParameterList()
        for _ in self.blocks:
            left = nn.Parameter(torch.zeros(self.config.width, rank, device=device))
            right = nn.Parameter(torch.empty(rank, self.config.width, device=device))
            nn.init.kaiming_uniform_(right, a=5**0.5)
            self.nlt_mix_left.append(left)
            self.nlt_mix_right.append(right)
        self.nlt_mix_scale = float(alpha) / float(rank)

    def enable_lora(self, rank: int, alpha: float) -> None:
        for block in self.blocks:
            block.spectral.enable_lora(rank=rank, alpha=alpha)


class DirichletOperator(nn.Module):
    """Boundary-lifted BL-RPNG operator and isolated legacy representation."""

    def __init__(
        self,
        config: FNOConfig,
        representation: str = "lifted",
        *,
        field_scales: tuple[float, ...] | None = None,
        gauge_channels: tuple[int, ...] = (),
        input_feature_mode: str = "raw",
        raw_condition_channels: int | None = None,
        diffusion_u: float = 2e-5,
        diffusion_v: float = 1e-5,
    ):
        super().__init__()
        if representation not in {
            "lifted", "raw_legacy", "boundary_augmented", "lift_conditioned"
        }:
            raise ValueError(f"Unknown representation: {representation}")
        if representation == "lifted":
            base_channels = 2 * config.out_channels + 2
        elif representation in {"boundary_augmented", "lift_conditioned"}:
            base_channels = 3 * config.out_channels + 2
        else:
            base_channels = config.out_channels + 2
        expected_channels = base_channels + int(config.condition_channels)
        if config.in_channels != expected_channels:
            raise ValueError(
                f"{representation} representation requires {expected_channels} input channels."
            )
        self.config = config
        self.representation = representation
        self.field_scales = tuple(float(value) for value in (field_scales or (1.0,) * config.out_channels))
        if len(self.field_scales) != config.out_channels or any(value <= 0 for value in self.field_scales):
            raise ValueError("field_scales must contain one positive value per output field.")
        self.gauge_channels = tuple(int(value) for value in gauge_channels)
        if any(value < 0 or value >= config.out_channels for value in self.gauge_channels):
            raise ValueError("gauge_channels contain an invalid field index.")
        if input_feature_mode not in {
            "raw", "gray_scott_coefficients", "gray_scott_reaction"
        }:
            raise ValueError(f"Unknown input_feature_mode: {input_feature_mode}")
        self.input_feature_mode = input_feature_mode
        self.raw_condition_channels = int(
            config.condition_channels
            if raw_condition_channels is None else raw_condition_channels
        )
        self.diffusion_u = float(diffusion_u)
        self.diffusion_v = float(diffusion_v)
        if input_feature_mode == "gray_scott_coefficients":
            if config.out_channels != 2 or self.raw_condition_channels < 2:
                raise ValueError("Gray--Scott coefficient features require two fields and F/K inputs.")
            if config.condition_channels != 4:
                raise ValueError("gray_scott_coefficients produces four condition channels.")
        if input_feature_mode == "gray_scott_reaction":
            if config.out_channels != 2 or self.raw_condition_channels < 2:
                raise ValueError("Gray--Scott reaction features require two fields and F/K inputs.")
            if config.condition_channels != 6:
                raise ValueError("gray_scott_reaction produces six condition channels.")
        self.fno = FNO2d(config)

    def _scales(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(self.field_scales).view(1, -1, 1, 1)

    def _apply_constraints(
        self,
        value: torch.Tensor,
        boundary_lift: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        constrained = torch.where(mask, boundary_lift, value)
        if self.gauge_channels:
            constrained = constrained.clone()
            for channel in self.gauge_channels:
                unconstrained = ~mask[:, channel : channel + 1]
                count = unconstrained.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
                mean = (constrained[:, channel : channel + 1] * unconstrained).sum(
                    dim=(-2, -1), keepdim=True
                ) / count
                constrained[:, channel : channel + 1] = torch.where(
                    unconstrained,
                    constrained[:, channel : channel + 1] - mean,
                    constrained[:, channel : channel + 1],
                )
        return constrained

    def forward(
        self,
        x: torch.Tensor,
        boundary_lift: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = boundary_mask.bool()
        expected_x_channels = self.config.out_channels + 2 + self.raw_condition_channels
        if x.shape[1] != expected_x_channels:
            raise ValueError(
                f"state input has {x.shape[1]} channels, expected {expected_x_channels} "
                "([state, grid_x, grid_y, optional conditions])."
            )
        if self.input_feature_mode != "raw":
            fields = self.config.out_channels
            state = x[:, :fields]
            grid = x[:, fields : fields + 2]
            raw = x[:, fields + 2 :]
            feed = raw[:, 0:1]
            kill = raw[:, 1:2]
            diffusion_u = torch.full_like(feed, self.diffusion_u / 2e-5)
            diffusion_v = torch.full_like(kill, self.diffusion_v / 1e-5)
            physical = [feed / 0.05, kill / 0.07, diffusion_u, diffusion_v]
            if self.input_feature_mode == "gray_scott_reaction":
                u, v = state[:, 0:1], state[:, 1:2]
                reaction_u = (-u * v.square() + feed * (1.0 - u)) / 0.05
                reaction_v = (u * v.square() - (feed + kill) * v) / 0.05
                physical.extend((reaction_u, reaction_v))
            x = torch.cat((state, grid, *physical), dim=1)
        if mask.shape[1] == 1 and self.config.out_channels > 1:
            mask = mask.expand(-1, self.config.out_channels, -1, -1)
        if mask.shape[1] != self.config.out_channels:
            raise ValueError("constraint mask must have one channel per field or one shared channel.")
        if self.representation == "lifted":
            fields = self.config.out_channels
            scales = self._scales(x)
            lifted_input = torch.cat(
                (
                    (x[:, :fields] - boundary_lift) / scales,
                    boundary_lift / scales,
                    x[:, fields : fields + 2],
                    x[:, fields + 2 :],
                ),
                dim=1,
            )
            residual, probes = self.fno(lifted_input)
            residual = residual * scales
            prediction = self._apply_constraints(residual + boundary_lift, boundary_lift, mask)
            probes = {
                name: (value * scales).masked_fill(mask, 0.0)
                for name, value in probes.items()
            }
            if self.gauge_channels:
                for name, value in probes.items():
                    value = value.clone()
                    for channel in self.gauge_channels:
                        value[:, channel : channel + 1] -= value[:, channel : channel + 1].mean(
                            dim=(-2, -1), keepdim=True
                        )
                    probes[name] = value
            return prediction, probes
        if self.representation in {"boundary_augmented", "lift_conditioned"}:
            fields = self.config.out_channels
            scales = self._scales(x)
            boundary_feature = (
                torch.where(mask, boundary_lift, torch.zeros_like(boundary_lift))
                if self.representation == "boundary_augmented"
                else boundary_lift
            )
            conditioned_input = torch.cat(
                (
                    x[:, :fields] / scales,
                    boundary_feature / scales,
                    mask.to(x.dtype),
                    x[:, fields : fields + 2],
                    x[:, fields + 2 :],
                ),
                dim=1,
            )
            prediction, probes = self.fno(conditioned_input)
            prediction = self._apply_constraints(
                prediction * scales, boundary_lift, mask
            )
            probes = {
                name: (value * scales).masked_fill(mask, 0.0)
                for name, value in probes.items()
            }
            return prediction, probes
        prediction, probes = self.fno(x)
        return prediction, probes

    def path_active_parameters(self, layer: int) -> list[nn.Parameter]:
        return self.fno.path_active_parameters(layer)

    def configure_adaptation(
        self,
        strategy: str,
        *,
        rank: int = 4,
        alpha: float = 4.0,
        train_last_blocks: int = 0,
    ) -> dict[str, int | str]:
        """Configure a task adapter and return auditable parameter counts."""
        if strategy not in {"full", "lora", "nlt", "nlt_mix", "nlt_lora"}:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")
        for parameter in self.parameters():
            parameter.requires_grad_(strategy == "full")
        if strategy == "lora":
            self.fno.enable_lora(rank=rank, alpha=alpha)
        elif strategy == "nlt":
            self.fno.enable_nlt()
        elif strategy == "nlt_mix":
            self.fno.enable_nlt_mixing(rank=rank, alpha=alpha)
        elif strategy == "nlt_lora":
            # NLT handles low-dimensional channel re-calibration, while the
            # spectral LoRA update can change the target-boundary frequency
            # response.  Both adapters are identity/zero initialized.
            self.fno.enable_nlt()
            self.fno.enable_lora(rank=rank, alpha=alpha)
        if strategy != "full":
            for parameter in self.fno.project.parameters():
                parameter.requires_grad_(True)
            for probe in self.fno.probes:
                for parameter in probe.parameters():
                    parameter.requires_grad_(True)
            if train_last_blocks:
                for block in self.fno.blocks[-int(train_last_blocks) :]:
                    for parameter in block.parameters():
                        parameter.requires_grad_(True)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"strategy": strategy, "trainable_parameters": trainable, "total_parameters": total}

    def model_config_dict(self) -> dict[str, int | str | list[float] | list[int]]:
        payload = self.config.to_dict()
        if any(value != 1.0 for value in self.field_scales):
            payload["field_scales"] = list(self.field_scales)
        if self.gauge_channels:
            payload["gauge_channels"] = list(self.gauge_channels)
        if self.input_feature_mode != "raw":
            payload["input_feature_mode"] = self.input_feature_mode
            payload["raw_condition_channels"] = self.raw_condition_channels
            payload["diffusion_u"] = self.diffusion_u
            payload["diffusion_v"] = self.diffusion_v
        return payload


def build_model(model_config: dict, representation: str = "lifted") -> DirichletOperator:
    config = FNOConfig(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        modes1=int(model_config["modes1"]),
        modes2=int(model_config["modes2"]),
        width=int(model_config["width"]),
        layers=int(model_config["layers"]),
        probe_mode=str(model_config.get("probe_mode", "per_layer")),
        condition_channels=int(model_config.get("condition_channels", 0)),
    )
    return DirichletOperator(
        config,
        representation=representation,
        field_scales=tuple(float(value) for value in model_config.get("field_scales", [1.0] * config.out_channels)),
        gauge_channels=tuple(int(value) for value in model_config.get("gauge_channels", [])),
        input_feature_mode=str(model_config.get("input_feature_mode", "raw")),
        raw_condition_channels=int(model_config.get("raw_condition_channels", config.condition_channels)),
        diffusion_u=float(model_config.get("diffusion_u", 2e-5)),
        diffusion_v=float(model_config.get("diffusion_v", 1e-5)),
    )
