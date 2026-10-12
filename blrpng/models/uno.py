from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blrpng.models.fno import DirichletOperator, FNOBlock, FNOConfig


class UNO2d(nn.Module):
    """Compact U-shaped neural operator used as the architecture-transfer check."""

    def __init__(self, config: FNOConfig):
        super().__init__()
        if config.layers != 5:
            raise ValueError("The independent UNO check uses exactly five operator blocks.")
        self.config = config
        width = config.width
        self.lift = nn.Conv2d(config.in_channels, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock(width, config.modes1, config.modes2),
            FNOBlock(width, max(config.modes1 // 2, 1), max(config.modes2 // 2, 1)),
            FNOBlock(width, max(config.modes1 // 4, 1), max(config.modes2 // 4, 1)),
            FNOBlock(width, max(config.modes1 // 2, 1), max(config.modes2 // 2, 1)),
            FNOBlock(width, config.modes1, config.modes2),
        ])
        self.skip1 = nn.Conv2d(2 * width, width, 1)
        self.skip0 = nn.Conv2d(2 * width, width, 1)
        self.probes = nn.ModuleList([nn.Conv2d(width, config.out_channels, 1) for _ in range(5)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(),
            nn.Conv2d(2 * width, config.out_channels, 1),
        )

    def enable_nlt(self) -> None:
        self.nlt_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, self.config.width, 1, 1)) for _ in self.blocks
        ])
        self.nlt_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.config.width, 1, 1)) for _ in self.blocks
        ])

    def _adapt(self, value: torch.Tensor, index: int) -> torch.Tensor:
        if hasattr(self, "nlt_scales"):
            return value * self.nlt_scales[index] + self.nlt_biases[index]
        return value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probes = {}
        h0 = self._adapt(self.blocks[0](self.lift(x)), 0)
        probes["layer0:hook0"] = self.probes[0](h0)
        h1 = self._adapt(self.blocks[1](F.avg_pool2d(h0, 2)), 1)
        probes["layer1:hook0"] = F.interpolate(self.probes[1](h1), size=h0.shape[-2:], mode="bilinear", align_corners=True)
        h2 = self._adapt(self.blocks[2](F.avg_pool2d(h1, 2)), 2)
        probes["layer2:hook0"] = F.interpolate(self.probes[2](h2), size=h0.shape[-2:], mode="bilinear", align_corners=True)
        up1 = F.interpolate(h2, size=h1.shape[-2:], mode="bilinear", align_corners=True)
        h3 = self._adapt(self.blocks[3](self.skip1(torch.cat((up1, h1), dim=1))), 3)
        probes["layer3:hook0"] = F.interpolate(self.probes[3](h3), size=h0.shape[-2:], mode="bilinear", align_corners=True)
        up0 = F.interpolate(h3, size=h0.shape[-2:], mode="bilinear", align_corners=True)
        h4 = self._adapt(self.blocks[4](self.skip0(torch.cat((up0, h0), dim=1)), activate=False), 4)
        probes["layer4:hook0"] = self.probes[4](h4)
        return self.project(h4), probes

    def path_active_parameters(self, layer: int) -> list[nn.Parameter]:
        if not 0 <= layer < 5:
            raise ValueError("UNO path layer is outside [0,4].")
        parameters: list[nn.Parameter] = []
        modules: list[nn.Module] = [self.lift, *self.blocks[: layer + 1]]
        if layer >= 3:
            modules.append(self.skip1)
        if layer >= 4:
            modules.append(self.skip0)
        for module in modules:
            parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
        if hasattr(self, "nlt_scales"):
            parameters.extend(parameter for parameter in self.nlt_scales[: layer + 1] if parameter.requires_grad)
            parameters.extend(parameter for parameter in self.nlt_biases[: layer + 1] if parameter.requires_grad)
        return list({id(parameter): parameter for parameter in parameters}.values())


class DirichletUNO(DirichletOperator):
    def __init__(self, config: FNOConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.fno = UNO2d(config)

    def configure_adaptation(self, strategy: str, **_: object) -> dict[str, int | str]:
        if strategy not in {"full", "nlt"}:
            raise ValueError("The UNO architecture check supports full or NLT adaptation.")
        for parameter in self.parameters():
            parameter.requires_grad_(strategy == "full")
        if strategy == "nlt":
            self.fno.enable_nlt()
            for parameter in self.fno.project.parameters():
                parameter.requires_grad_(True)
            for probe in self.fno.probes:
                for parameter in probe.parameters():
                    parameter.requires_grad_(True)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"strategy": strategy, "trainable_parameters": trainable, "total_parameters": total}
