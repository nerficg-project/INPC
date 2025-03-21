# -- coding: utf-8 --

"""
INPC/utils.py: Utility functions for INPC.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from Cameras.Perspective import PerspectiveCamera


@dataclass(frozen=True)
class LRDecayPolicy(object):
    """Allows for flexible definition of a decay policy for a learning rate."""
    lr_init: float = 1.0
    lr_final: float = 1.0
    lr_delay_steps: int = 0
    lr_delay_mult: float = 1.0
    max_steps: int = 1000000

    # taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py#L78
    def __call__(self, iteration: int) -> float:
        """Calculates learning rate for the given iteration."""
        if iteration < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if self.lr_delay_steps > 0 and iteration < self.lr_delay_steps:
            # reverse cosine delay
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(iteration / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(iteration / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp


class TruncExp(torch.autograd.Function):
    """
    Truncated exponential function adapted from Instant-NGP for numerical stability.
    https://github.com/NVlabs/instant-ngp/blob/afe4057bb49e23926b94c07f07ff2ee113925007/include/neural-graphics-primitives/nerf_device.cuh#L250
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        return grad * torch.exp(x.clamp(-15.0, 15.0))


# @torch.compile  # TORCH_COMPILE_NOTE: uncommenting this makes things slightly faster
def feature_to_opacity(features: torch.Tensor) -> torch.Tensor:
    """Converts logspace volume density to opacity in [0, 1]."""
    return 1.0 - torch.exp(-TruncExp.apply(features))


# @torch.compile  # TORCH_COMPILE_NOTE: uncommenting this makes things slightly faster
def spherical_contraction(positions: torch.Tensor) -> torch.Tensor:
    """
    Normalize positions to [0, 1] using the spherical contraction from Mip-NeRF360. Adapted from:
    https://github.com/jonbarron/camp_zipnerf/blob/16206bd88f37d5c727976557abfbd9b4fa28bbe1/internal/coord.py#L26.
    """
    length_squared = positions.square().sum(dim=-1, keepdim=True).clamp_min(1.0)
    scale = length_squared.sqrt().mul(2).sub(1).div(length_squared).mul(0.25)
    return positions.mul(scale).add(0.5).clamp(0.0, 1.0)


def halton_base(index: int, base: int) -> float:
    """Generates a Halton sequence element for the given index and base."""
    result = 0
    f = 1 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base
    return result


def generate_halton_sequence(n: int, d: int) -> torch.Tensor:
    """Generates a Halton sequence of length n in d dimensions."""
    primes = [2, 3, 5]
    halton_points = torch.zeros((n, d))
    for i in range(n):
        for j in range(d):
            halton_points[i][j] = halton_base(i + 1, primes[j])
    return halton_points


def compute_halton_coords(counts: torch.Tensor, cell_indices_samples: torch.Tensor) -> torch.Tensor:
    """Computes Halton coordinates for the given cell indices and sample counts."""
    offsets = torch.cumsum(counts, dim=0) - counts  # exclusive sum
    halton_indices = torch.arange(counts.sum().item(), dtype=torch.int32, device=cell_indices_samples.device)
    halton_indices = halton_indices - offsets[cell_indices_samples]
    halton_coords = generate_halton_sequence(halton_indices.max().item() + 1, 3)[halton_indices]
    return halton_coords


@dataclass()
class PaddingInfo:
    """Information about padding applied to a camera."""
    original_width: int
    original_height: int
    original_principal_offset_x: float
    original_principal_offset_y: float
    roi_slices: tuple[slice, slice] = (slice(None), slice(None))

    def unapply(self, camera: 'PerspectiveCamera') -> None:
        """Removes padding from the given camera."""
        camera.properties.width = self.original_width
        camera.properties.height = self.original_height
        camera.properties.principal_offset_x = self.original_principal_offset_x
        camera.properties.principal_offset_y = self.original_principal_offset_y


def adjust_for_unet(camera: 'PerspectiveCamera') -> PaddingInfo:
    """Adjusts the camera properties to prevent the need for padding inside a U-Net with two downsampling steps."""
    padding_info = PaddingInfo(
        camera.properties.width,
        camera.properties.height,
        camera.properties.principal_offset_x,
        camera.properties.principal_offset_y
    )
    left, right, top, bottom = 0, 0, 0, 0
    if camera.properties.width % 4 != 0:
        padding = 4 - camera.properties.width % 4
        camera.properties.width += padding
        if padding != 2:
            camera.properties.principal_offset_x -= 0.5
        left = padding // 2
        right = padding - left
    if camera.properties.height % 4 != 0:
        padding = 4 - camera.properties.height % 4
        camera.properties.height += padding
        if padding != 2:
            camera.properties.principal_offset_y -= 0.5
        top = padding // 2
        bottom = padding - top
    padding_info.roi_slices = slice(top, camera.properties.height - bottom), slice(left, camera.properties.width - right)
    return padding_info


@dataclass
class PreextractedPointCloud:
    """Helper class for pre-extracted point clouds."""
    _initialized: bool = False
    positions: torch.Tensor | None = None
    features: torch.Tensor | None = None
    opacities: torch.Tensor | None = None

    def set(self, positions: torch.Tensor, features: torch.Tensor, opacities: torch.Tensor) -> None:
        """Sets the pre-extracted point cloud."""
        self.positions = positions
        self.features = features
        self.opacities = opacities
        self._initialized = True

    def clear(self) -> None:
        """Clears the pre-extracted point cloud."""
        self.positions = None
        self.features = None
        self.opacities = None
        self._initialized = False

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the pre-extracted point cloud."""
        return self.positions, self.features, self.opacities

    @property
    def is_set(self) -> bool:
        """Returns whether point cloud is currently initialized."""
        return self._initialized
