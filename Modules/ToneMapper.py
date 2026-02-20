"""
INPC/Modules/ToneMapper.py: Implementation of a differentiable tone mapping module.
adapted from ADOP: https://github.com/darglein/ADOP
"""

import math
import torch

from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger
from Optim.lr_utils import LRDecayPolicy


class ToneMapper(torch.nn.Module):
    """Tone mapping module capable of learning a response function and per-view exposure."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        # init response params
        n_params = 27
        response = torch.linspace(0.0, 1.0, n_params).pow(0.4545454681)
        response = response / response[-1]
        response = response[None, None, None, :].repeat(3, 1, 1, 1)
        self.register_parameter('response_params', torch.nn.Parameter(response))
        smoothness_factor = 1.0e-5
        self.response_smoothness_factor = n_params * math.sqrt(smoothness_factor)

    @torch.no_grad()
    def apply_response_constraints(self) -> None:
        """Apply constraints to the response parameters."""
        self.response_params.data[..., 0] = 0.0
        self.response_params.data[..., -1] = 1.0
        # if (self.response_params.data[..., 1:-1] <= 0.0).any() or (self.response_params.data[..., 1:-1] >= 1.0).any():
        #     Logger.log_warning('response parameters are outside [0, 1] range')
        self.response_params.data.clamp_(0.0, 1.0)  # FIXME: would love to have a more elegant solution here

    def setup_exposure_params(self, dataset: BaseDataset) -> None:
        """Set up the exposure parameters."""
        n_cameras = len(dataset.test()) + len(dataset.train())
        exposure_params = torch.zeros(n_cameras)
        self.register_parameter('exposure_params', torch.nn.Parameter(exposure_params))

    def apply_exposure(self, image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Apply exposure correction to the image."""
        return image * torch.exp2(-self.exposure_params[view_idx])

    def unapply_exposure(self, image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """Inverse of apply_exposure."""
        return image * torch.exp2(self.exposure_params[view_idx].to(image.device, image.dtype))

    def apply_response(self, image: torch.Tensor) -> torch.Tensor:
        """Apply response correction to the image."""
        # FIXME: fusing the forward/backward pass would save some time (esp. backward)
        leak_add = None
        if self.training:
            clamp_low = image < 0.0
            clamp_high = image > 1.0
            leak_add = (image * 0.01) * clamp_low
            leak_add += (-0.01 / image.clamp_min(1.0).sqrt() + 0.01) * clamp_high
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

    def unapply_response(self, image: torch.Tensor) -> torch.Tensor:
        """Inverse of apply_response."""
        n_params = self.response_params.shape[-1]
        y_vals = self.response_params[:, 0, 0, :].to(image.device, image.dtype)
        delta_x = 2.0 / (n_params - 1)
        height, width = image.shape[1:]
        result = torch.empty_like(image)
        for ch in range(3):
            image_flat_ch = image[ch].reshape(-1)
            vals = y_vals[ch]
            idx = torch.searchsorted(vals, image_flat_ch, right=True).clamp(min=1, max=n_params - 1)
            y0 = vals[idx - 1]
            y1 = vals[idx]
            y_ranges = y1 - y0
            if (y_ranges <= 0).any():
                Logger.log_warning(f'response in channel {ch} is not strictly monotonically increasing -> inverse might not work correctly')
                y_ranges = y_ranges.clamp_min(1.0e-12)
            t = (image_flat_ch - y0) / y_ranges
            x0 = -1 + (idx - 1).to(image.dtype) * delta_x
            x = x0 + t * delta_x
            result[ch] = x.add(1.0).mul(0.5).reshape(height, width)
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
                lr_init=1.0e-4,
                lr_final=1.0e-4,
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

    def inverse_forward(self, image: torch.Tensor, view: View) -> torch.Tensor:
        """Apply inverse tone mapping to the input image."""
        image = self.unapply_response(image)
        image = self.unapply_exposure(image, view.global_frame_idx)  # TODO: handle invalid/unseen global_frame_idx
        return image

    def forward(self, image: torch.Tensor, view: View) -> torch.Tensor:
        """Apply tone mapping to the input image."""
        image = self.apply_exposure(image, view.global_frame_idx)  # TODO: handle invalid/unseen global_frame_idx
        image = self.apply_response(image)
        return image
