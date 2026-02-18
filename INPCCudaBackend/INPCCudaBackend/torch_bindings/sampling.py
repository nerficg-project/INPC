import torch

from Datasets.utils import View
from Cameras.Perspective import PerspectiveCamera

from INPCCudaBackend import _C


def compute_viewpoint_weights(
    centers: torch.Tensor,
    levels: torch.Tensor,
    weights: torch.Tensor,
    view: View,
    initial_size: float,
) -> torch.Tensor:
    if not isinstance(view.camera, PerspectiveCamera):
        raise NotImplementedError
    if view.camera.distortion is not None:
        raise NotImplementedError
    return _C.compute_viewpoint_weights_cuda(
        centers,
        levels,
        weights,
        view.w2c,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
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
        view: View,
        n_samples: int,
        initial_size: float,
    ) -> torch.Tensor:
        if not isinstance(view.camera, PerspectiveCamera):
            raise NotImplementedError
        if view.camera.distortion is not None:
            raise NotImplementedError
        return self.cuda_backend.generate_samples(
            centers,
            levels,
            weights,
            view.w2c,
            n_samples,
            view.camera.width,
            view.camera.height,
            view.camera.focal_x,
            view.camera.focal_y,
            view.camera.center_x,
            view.camera.center_y,
            view.camera.near_plane,
            view.camera.far_plane,
            initial_size,
        )

    def generate_expected_samples(
        self,
        centers: torch.Tensor,
        levels: torch.Tensor,
        weights: torch.Tensor,
        view: View,
        n_samples: int,
        n_multi: int,
        initial_size: float,
    ) -> torch.Tensor:
        if not isinstance(view.camera, PerspectiveCamera):
            raise NotImplementedError
        if view.camera.distortion is not None:
            raise NotImplementedError
        return self.cuda_backend.generate_expected_samples(
            centers,
            levels,
            weights,
            view.w2c,
            n_samples,
            n_multi,
            view.camera.width,
            view.camera.height,
            view.camera.focal_x,
            view.camera.focal_y,
            view.camera.center_x,
            view.camera.center_y,
            view.camera.near_plane,
            view.camera.far_plane,
            initial_size,
        )
