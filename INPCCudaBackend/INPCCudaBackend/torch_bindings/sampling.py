import torch

from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import IdentityDistortion
from INPCCudaBackend import _C


def compute_viewpoint_weights(
        centers: torch.Tensor,
        levels: torch.Tensor,
        weights: torch.Tensor,
        camera: PerspectiveCamera,
        initial_size: float,
) -> torch.Tensor:
    if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
        raise NotImplementedError
    return _C.compute_viewpoint_weights_cuda(
        centers,
        levels,
        weights,
        camera.properties.w2c,
        camera.properties.width,
        camera.properties.height,
        camera.properties.focal_x,
        camera.properties.focal_y,
        camera.properties.principal_offset_x,
        camera.properties.principal_offset_y,
        camera.near_plane,
        camera.far_plane,
        initial_size,
    )


class ProbabilityFieldSampler(torch.nn.Module):
    def __init__(
            self,
            seed: int,
    ) -> None:
        super().__init__()
        self.cuda_backend = _C.ProbabilityFieldSamplerCUDA(seed)

    def generate_samples(
            self,
            centers: torch.Tensor,
            levels: torch.Tensor,
            weights: torch.Tensor,
            camera: PerspectiveCamera,
            n_samples: int,
            initial_size: float,
    ) -> torch.Tensor:
        if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
            raise NotImplementedError
        return self.cuda_backend.generate_samples(
            centers,
            levels,
            weights,
            camera.properties.w2c,
            n_samples,
            camera.properties.width,
            camera.properties.height,
            camera.properties.focal_x,
            camera.properties.focal_y,
            camera.properties.principal_offset_x,
            camera.properties.principal_offset_y,
            camera.near_plane,
            camera.far_plane,
            initial_size,
        )

    def generate_expected_samples(
            self,
            centers: torch.Tensor,
            levels: torch.Tensor,
            weights: torch.Tensor,
            camera: PerspectiveCamera,
            n_samples: int,
            n_multi: int,
            initial_size: float,
    ) -> torch.Tensor:
        if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
            raise NotImplementedError
        return self.cuda_backend.generate_expected_samples(
            centers,
            levels,
            weights,
            camera.properties.w2c,
            n_samples,
            n_multi,
            camera.properties.width,
            camera.properties.height,
            camera.properties.focal_x,
            camera.properties.focal_y,
            camera.properties.principal_offset_x,
            camera.properties.principal_offset_y,
            camera.near_plane,
            camera.far_plane,
            initial_size,
        )
