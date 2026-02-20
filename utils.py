"""
INPC/utils.py: Utility functions for INPC.
"""

from dataclasses import dataclass
from typing import Any

import torch

from Cameras.Perspective import PerspectiveCamera


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


def feature_to_opacity(features: torch.Tensor) -> torch.Tensor:
    """Converts logspace volume density to opacity in [0, 1]."""
    return -torch.expm1(-TruncExp.apply(features))


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
    original_center_x: float
    original_center_y: float
    roi_slices: tuple[slice, slice] = (slice(None), slice(None))

    def unapply(self, camera: PerspectiveCamera) -> None:
        """Removes padding from the given camera."""
        camera.width = self.original_width
        camera.height = self.original_height
        camera.center_x = self.original_center_x
        camera.center_y = self.original_center_y


def adjust_for_unet(camera: PerspectiveCamera) -> PaddingInfo:
    """Adjusts the camera intrinsics to prevent the need for padding inside a U-Net with two downsampling steps."""
    padding_info = PaddingInfo(
        camera.width,
        camera.height,
        camera.center_x,
        camera.center_y
    )
    left, right, top, bottom = 0, 0, 0, 0
    if camera.width % 4 != 0:
        padding = 4 - camera.width % 4
        camera.width += padding
        camera.center_x += padding / 2
        if padding != 2:
            camera.center_x -= 0.5
        left = padding // 2
        right = padding - left
    if camera.height % 4 != 0:
        padding = 4 - camera.height % 4
        camera.height += padding
        camera.center_y += padding / 2
        if padding != 2:
            camera.center_y -= 0.5
        top = padding // 2
        bottom = padding - top
    padding_info.roi_slices = slice(top, camera.height - bottom), slice(left, camera.width - right)
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


@dataclass
class MultisamplingRingBuffer:
    """Helper class for multisampled INPC rendering."""
    size = 0
    positions: list[torch.Tensor] | list[None] | None = None
    features: list[torch.Tensor] | list[None] | None = None
    current_idx: int = 0
    read_idx: int = 0

    def append(self, positions: torch.Tensor, features: torch.Tensor, new_size: int) -> None:
        """Sets the current multisampling ring buffer."""
        if self.size != new_size:
            self.current_idx = 0
            self.size = new_size
            self.positions = [None] * new_size
            self.features = [None] * new_size
        self.positions[self.current_idx] = positions
        self.features[self.current_idx] = features
        self.current_idx += 1
        self.current_idx %= self.size
        self.read_idx = 0

    def get_next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the pre-extracted point cloud."""
        if self.positions[self.read_idx] is None:
            return self.positions[0], self.features[0]
        positions = self.positions[self.read_idx]
        features = self.features[self.read_idx]
        self.read_idx = self.read_idx + 1
        return positions, features
