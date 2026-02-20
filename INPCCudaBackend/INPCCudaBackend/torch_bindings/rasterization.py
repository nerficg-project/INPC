from typing import Any, NamedTuple
from enum import Enum

import torch
from torch.autograd.function import once_differentiable

from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger

from INPCCudaBackend import _C


class RasterizerMode(Enum):
    BILINEAR = 0
    GAUSSIAN = 1


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor
    cam_position: torch.Tensor
    mode: RasterizerMode
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    near_plane: float
    far_plane: float

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.mode.value,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
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
        image, alpha, blending_weights, per_primitive_buffers, per_tile_buffers, per_instance_buffers, n_instances, instance_primitive_indices_selector = _C.forward(
            positions,
            features,
            opacities,
            *rasterizer_settings.as_tuple(),
        )
        ctx.save_for_backward(
            image,
            alpha,
            positions,
            opacities,
            per_primitive_buffers,
            per_tile_buffers,
            per_instance_buffers,
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.n_instances = n_instances
        ctx.instance_primitive_indices_selector = instance_primitive_indices_selector
        ctx.mark_non_differentiable(blending_weights)
        ctx.set_materialize_grads(False)
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
            *ctx.rasterizer_settings.as_tuple(),
            ctx.n_instances,
            ctx.instance_primitive_indices_selector,
        )
        return (
            None,  # positions
            grad_features,
            grad_opacities,
            None,  # rasterizer_settings
        )


class INPCRasterizer(torch.nn.Module):
    @staticmethod
    def extract_settings(view: View, mode: int) -> RasterizerSettings:
        if not isinstance(view.camera, PerspectiveCamera):
            raise NotImplementedError
        if view.camera.distortion is not None:
            Logger.log_warning('rasterizer ignores all distortion parameters')
        try:
            rasterizer_mode = RasterizerMode(mode)
        except ValueError:
            Logger.log_warning(f'invalid rasterizer mode {mode}, defaulting to BILINEAR')
            rasterizer_mode = RasterizerMode.BILINEAR
        return RasterizerSettings(
            view.w2c,
            view.position,
            rasterizer_mode,
            view.camera.width,
            view.camera.height,
            view.camera.focal_x,
            view.camera.focal_y,
            view.camera.center_x,
            view.camera.center_y,
            view.camera.near_plane,
            view.camera.far_plane,
        )

    @staticmethod
    def extract_base_sigma_world(camera: BaseCamera) -> float:
        if not isinstance(camera, PerspectiveCamera):
            raise NotImplementedError
        return 0.05 / camera.focal_y  # results in screen-space sigma of 5px at default near plane (0.01)

    def forward(
        self,
        view: View,
        mode: int,
        positions: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _Rasterize.apply(
            positions,
            features,
            opacities,
            self.extract_settings(view, mode),
        )

    def render(
        self,
        view: View,
        mode: int,
        positions: torch.Tensor,
        features_raw: torch.Tensor,
        sigma_world_scale: float,
        sigma_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, depth, alpha = _C.render(
            positions,
            features_raw,
            *self.extract_settings(view, mode).as_tuple(),
            sigma_world_scale * self.extract_base_sigma_world(view.camera),
            sigma_cutoff,
        )
        return image, depth, alpha

    def render_preextracted(
        self,
        view: View,
        mode: int,
        positions: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor | None = None,
        sigma_world_scale: float = 1.0,
        sigma_cutoff: float = 3.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, depth, alpha = _C.render_preextracted(
            positions,
            features,
            opacities,
            torch.empty(0) if scales is None else scales,
            *self.extract_settings(view, mode).as_tuple(),
            sigma_world_scale * self.extract_base_sigma_world(view.camera),
            sigma_cutoff,
        )
        return image, depth, alpha
