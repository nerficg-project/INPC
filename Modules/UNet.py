"""
INPC/Modules/UNet.py: U-Net implementation.
FFCResidualBlock implementation is based on FFC: https://github.com/pkumivision/FFC
"""

from pathlib import Path
import torch

import Framework
from Logging import Logger
from Optim.lr_utils import LRDecayPolicy


class DoubleConv(torch.nn.Module):
    """(convolution => activation) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.SiLU(inplace=True),
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
        # x has shape (B, C, H, W), i.e., (B, 96, H/4, W/4) for the default INPC architecture; let Wf = W/2+1
        x = x.float()
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')  # (B, C, H, Wf)
        B, C, H, Wf = ffted.shape
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (B, C, H, Wf, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, Wf)  # (B, C*2, H, Wf)
        ffted = self.conv(ffted)
        ffted = ffted.reshape(B, C, 2, H, Wf).permute(0, 1, 3, 4, 2)  # (B, C, H, Wf, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])  # (B, C, H, Wf)
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
        self.activation = torch.nn.SiLU(inplace=True)

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
        self.initialized_from_checkpoint = False

    def set_weights(self, checkpoint_path: Path, exclude_residual_block: bool = True) -> None:
        """Overwrites the current weights with those from the given checkpoint, excluding residual block weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Optionally filter out keys that belong to residual_block
            if exclude_residual_block:
                checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('residual_block.')}
            self.load_state_dict(checkpoint, strict=False)
        except IOError as e:
            Logger.log_error(f'failed to load U-Net weights: "{e}"')
        self.to(Framework.config.GLOBAL.DEFAULT_DEVICE)
        self.initialized_from_checkpoint = True

    def save_weights(self, path: Path) -> None:
        """Saves the current weights in a '.pt' file."""
        try:
            torch.save(self.state_dict(), path)
        except IOError as e:
            Logger.log_error(f'failed to save U-Net weights: "{e}"')

    def get_optimizer_param_groups(self, max_iterations: int, lr_init: float = 3e-4, lr_final: float = 5e-5) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        lr_delay_steps = 0
        if self.initialized_from_checkpoint:
            lr_delay_steps = 12_500
            Logger.log_info(f'using pretrained weights for U-Net -> setting lr_delay_steps to {lr_delay_steps:,}')
        schedulers = [LRDecayPolicy(
            lr_init=lr_init,
            lr_final=lr_final,
            lr_delay_steps=lr_delay_steps,
            lr_delay_mult=1e-4,
            max_steps=max_iterations
        )]
        return param_groups, schedulers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the U-Net for the given input of shape ([B, ]C, H, W)."""
        # add batch dimension if not present
        if batch_dim_added := x.dim() == 3:
            x = x[None]
        x0 = self.start(x)
        x1 = self.down1(x0)
        x = self.down2(x1)
        x = self.residual_block(x)
        x = self.up1(x, x1)
        x = self.up2(x, x0)
        x = self.end_block(x)
        x = x * 0.5 + 0.5
        return x[0] if batch_dim_added else x
