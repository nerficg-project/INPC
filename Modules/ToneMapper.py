"""
INPC/Modules/ToneMapper.py: Implementation of a differentiable tone mapping module.
adapted from ADOP: https://github.com/darglein/ADOP
"""

import math
import torch

from Datasets.Base import BaseDataset
from Datasets.utils import View
from Optim.lr_utils import LRDecayPolicy


class ToneMapper(torch.nn.Module):
    """Tone mapping module capable of learning a response function and per-view exposure."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        # init response params
        n_params = 25
        response = torch.linspace(0.0, 1.0, n_params).pow(0.4545454681)
        response = response / response[-1]
        response = response[None, None, None, :].repeat(3, 1, 1, 1)
        self.register_parameter('response_params', torch.nn.Parameter(response))
        smoothness_factor = 1.0e-5
        self.response_smoothness_factor = n_params * math.sqrt(smoothness_factor)

    def setup_exposure_params(self, dataset: BaseDataset) -> None:
        """Set up the exposure parameters."""
        n_cameras = len(dataset.test()) + len(dataset.train())
        exposure_params = torch.zeros(n_cameras)
        self.register_parameter('exposure_params', torch.nn.Parameter(exposure_params))

    def apply_exposure(self, image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Apply exposure correction to the image."""
        return image * torch.exp2(-self.exposure_params[view_idx])

    def apply_response(self, image: torch.Tensor) -> torch.Tensor:
        """Apply response correction to the image."""
        leak_add = None
        if self.training:
            clamp_low = image < 0.0
            clamp_high = image > 1.0
            leak_add = (image * 0.01) * clamp_low
            leak_add += (-0.01 / image.abs().add(1.0e-4).sqrt() + 0.01) * clamp_high
        x = torch.empty(*image.shape, 2, dtype=image.dtype, device=image.device)
        x[..., 0] = image * 2.0 - 1.0
        x[..., 1].zero_()
        result = torch.nn.functional.grid_sample(
            input=self.response_params, grid=x,
            align_corners=True, mode='bilinear', padding_mode='border'
        ).squeeze()
        if leak_add is not None:
            result += leak_add
        return result

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [
            {'params': [self.exposure_params], 'lr': 1.0},
            {'params': [self.response_params], 'lr': 1.0},
        ]
        schedulers = [
            LRDecayPolicy(
                lr_init=5.0e-4,
                lr_final=5.0e-4,
                lr_delay_steps=5000,
                lr_delay_mult=0.01,
                max_steps=max_iterations),
            LRDecayPolicy(
                lr_init=1.0e-3,
                lr_final=1.0e-3,
                lr_delay_steps=5000,
                lr_delay_mult=0.01,
                max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def response_loss(self) -> torch.Tensor:
        """Calculate the response loss."""
        low = self.response_params[..., :-2]
        up = self.response_params[..., 2:]
        target = self.response_params.clone()
        target[..., 0] = 0.0
        target[..., 1:-1] = (up + low) * 0.5
        return torch.nn.functional.mse_loss(
            self.response_params * self.response_smoothness_factor,
            target * self.response_smoothness_factor,
            reduction='sum'
        )

    @torch.no_grad()
    def interpolate_testset_exposures(self, dataset: 'BaseDataset') -> None:
        """Interpolate the exposure parameters for the test set."""
        dataset.test()
        n_cameras = len(self.exposure_params)
        for view in dataset:
            idx = view.global_frame_idx
            if idx == 0:
                self.exposure_params[idx] = self.exposure_params[1]
            elif idx == n_cameras - 1:
                self.exposure_params[idx] = self.exposure_params[n_cameras - 2]
            else:
                self.exposure_params[idx] = 0.5 * (self.exposure_params[idx - 1] + self.exposure_params[idx + 1])

    def forward(self, image: torch.Tensor, view: View) -> torch.Tensor:
        """Apply tone mapping to the input image."""
        image = self.apply_exposure(image, view.global_frame_idx)  # TODO: handle invalid/unseen global_frame_idx
        image = self.apply_response(image)
        return image
