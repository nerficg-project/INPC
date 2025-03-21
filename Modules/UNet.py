# -- coding: utf-8 --

"""
INPC/Modules/UNet.py: U-Net implementation.
FFCResidualBlock implementation is based on FFC: https://github.com/pkumivision/FFC
"""

import torch

from Methods.INPC.utils import LRDecayPolicy


class DoubleConv(torch.nn.Module):
    """(convolution => activation) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(torch.nn.Module):
    """Downscaling with average pooling into convolution primitive"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down_conv = torch.nn.Sequential(
            torch.nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_conv(x)


class UpBlock(torch.nn.Module):
    """Upscaling into convolution primitive"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class FourierUnit(torch.nn.Module):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_features, n_features, 1, bias=False),
            torch.nn.BatchNorm2d(n_features),
            torch.nn.ReLU(inplace=True)
        )

    @torch.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (1, C, H, W), i.e., (1, 96, H/4, W/4) for the default INPC architecture
        x = x.float()
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')  # (B, C, H, W/2+1)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (B, C, H, W/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, 2, H, W/2+1)
        ffted = ffted.view((1, -1,) + ffted.size()[3:])  # (B, C*2, H, W/2+1)
        ffted = self.conv(ffted)
        ffted = ffted.view((1, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])  # (B, C, H, W/2+1)
        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm='ortho')  # (B, C, H, W)
        return output


class SpectralTransform(torch.nn.Module):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_features, n_features // 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(n_features // 2),
            torch.nn.ReLU(inplace=True)
        )
        self.fourier_unit = FourierUnit(n_features)
        self.conv2 = torch.nn.Conv2d(n_features // 2, n_features, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        output = self.fourier_unit(x)
        output = self.conv2(x + output)
        return output


class FFC(torch.nn.Module):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        n_features_local = n_features // 4
        n_features_global = n_features - n_features_local
        self.conv_local_to_local = torch.nn.Conv2d(n_features_local, n_features_local, 3, padding=1, bias=False)
        self.conv_local_to_global = torch.nn.Conv2d(n_features_local, n_features_global, 3, padding=1, bias=False)
        self.conv_global_to_local = torch.nn.Conv2d(n_features_global, n_features_local, 3, padding=1, bias=False)
        self.conv_global_to_global = SpectralTransform(n_features_global)
        self.activation = torch.nn.GELU()

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_xl = self.conv_local_to_local(x_local) + self.conv_global_to_local(x_global)
        out_xg = self.conv_local_to_global(x_local) + self.conv_global_to_global(x_global)
        return self.activation(out_xl), self.activation(out_xg)


class FFCResidualBlock(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.ffc1 = FFC(n_features)
        self.ffc2 = FFC(n_features)
        self.n_features_local = n_features // 4
        self.n_features_global = n_features - self.n_features_local

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local, x_global = torch.split(x, [self.n_features_local, self.n_features_global], dim=1)
        x_local, x_global = self.ffc1(x_local, x_global)
        x_local, x_global = self.ffc2(x_local, x_global)
        output = torch.cat((x_local, x_global), dim=1)
        return output + x


class UNet(torch.nn.Module):
    """U-Net model for hole-filling."""

    def __init__(self, use_ffc_block: bool) -> None:
        super().__init__()
        self.start = DoubleConv(4, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.residual_block = FFCResidualBlock(256) if use_ffc_block else torch.nn.Identity()
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.end_block = torch.nn.Conv2d(64, 3, kernel_size=1)

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=3.0e-4,
            lr_final=5.0e-5,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=max_iterations
        )]
        return param_groups, schedulers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the U-Net for the given input of shape (C, H, W)."""
        x0 = self.start(x.unsqueeze(0))
        x1 = self.down1(x0)
        x = self.down2(x1)
        x = self.residual_block(x)
        x = self.up1(x, x1)
        x = self.up2(x, x0)
        x = self.end_block(x)
        x = x * 0.5 + 0.5
        return x.squeeze()
