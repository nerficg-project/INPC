from typing import Any, NamedTuple

import torch
from torch.autograd.function import once_differentiable

from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import IdentityDistortion
from INPCCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor
    cam_position: torch.Tensor
    width: int
    height: int
    focal_x: float
    focal_y: float
    principal_offset_x: float
    principal_offset_y: float
    near_plane: float
    far_plane: float

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.principal_offset_x,
            self.principal_offset_y,
            self.near_plane,
            self.far_plane,
        )

class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            positions: torch.Tensor,
            features: torch.Tensor,
            opacities: torch.Tensor,
            rasterizer_settings: RasterizerSettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, alpha, blending_weights, per_point_buffers, per_pixel_buffers, fragment_point_indices_selector = _C.forward(
            positions,
            features,
            opacities,
            *rasterizer_settings.as_tuple(),
        )
        ctx.save_for_backward(image, alpha, positions, opacities, per_point_buffers, per_pixel_buffers, rasterizer_settings.cam_position)
        ctx.fragment_point_indices_selector = fragment_point_indices_selector
        ctx.mark_non_differentiable(blending_weights)
        return image, alpha, blending_weights

    @staticmethod
    @once_differentiable
    def backward(
            ctx: Any,
            grad_image: torch.Tensor,
            grad_alpha: torch.Tensor,
            _
    ) -> tuple[None, torch.Tensor, torch.Tensor, None]:
        grad_features, grad_opacities = _C.backward(
            grad_image,
            grad_alpha,
            *ctx.saved_tensors,
            ctx.fragment_point_indices_selector,
        )
        return (
            None,  # positions
            grad_features,
            grad_opacities,
            None,  # rasterizer_settings
        )


class INPCRasterizer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_settings(camera: PerspectiveCamera) -> RasterizerSettings:
        if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
            raise NotImplementedError
        return RasterizerSettings(
            camera.properties.w2c,
            camera.properties.T,
            camera.properties.width,
            camera.properties.height,
            camera.properties.focal_x,
            camera.properties.focal_y,
            camera.properties.principal_offset_x,
            camera.properties.principal_offset_y,
            camera.near_plane,
            camera.far_plane,
        )

    def forward(
            self,
            camera: PerspectiveCamera,
            positions: torch.Tensor,
            features: torch.Tensor,
            opacities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _Rasterize.apply(positions, features, opacities, self.extract_settings(camera))

    def render(
            self,
            camera: PerspectiveCamera,
            positions: torch.Tensor,
            features_raw: torch.Tensor,
            bg_image: torch.Tensor,
            n_multisamples: int,
    ) -> torch.Tensor:
        return _C.render(
            positions,
            features_raw,
            bg_image,
            *self.extract_settings(camera).as_tuple(),
            n_multisamples
        )

    def render_preextracted(
            self,
            camera: PerspectiveCamera,
            positions: torch.Tensor,
            features: torch.Tensor,
            opacities: torch.Tensor,
            bg_image: torch.Tensor,
    ) -> torch.Tensor:
        return _C.render_preextracted(
            positions,
            features,
            opacities,
            bg_image,
            *self.extract_settings(camera).as_tuple(),
        )
