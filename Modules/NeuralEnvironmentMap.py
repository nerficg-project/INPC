"""
INPC/Modules/NeuralEnvironmentMap.py: Neural environment map module.
"""

import torch

import Framework
from Datasets.utils import View
from Optim.lr_utils import LRDecayPolicy
import Thirdparty.TinyCudaNN as tcnn


class NeuralEnvironmentMap(torch.nn.Module):
    """Environment map using spherical harmonics encoding and an MLP."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.net_with_encoding = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=4,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 128,
                'n_hidden_layers': 4,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.net_with_encoding.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-3,
            lr_final=1.0e-3 / 30.0,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=max_iterations)
        ]
        return param_groups, schedulers

    def forward(self, view: View) -> torch.Tensor:
        """Returns environment map values for the given view."""
        view_dirs_all = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_dirs_all = torch.nn.functional.normalize(view_dirs_all, p=2, dim=-1)
        view_dirs_all.mul_(0.5).add_(0.5)  # for compatibility with the SH encoding in tiny-cuda-nn
        return self.net_with_encoding(view_dirs_all).reshape(view.camera.height, view.camera.width, 4)
