"""
INPC/Modules/NeuralEnvironmentMap.py: Neural environment map module.
"""

import torch

import Framework
from Logging import Logger
from Cameras.utils import directions_to_equirectangular_grid_coords
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
        self.register_buffer('distilled_texture', None)

    @torch.amp.autocast('cuda', enabled=False)
    def compute_ray_directions(self, view: View) -> torch.Tensor:
        """Returns the ray directions for the given camera."""
        # compute or retrieve cached local ray directions
        local_directions = view.camera.compute_local_ray_directions()
        # transform to world space
        directions = view.cam_to_world(local_directions, is_point=False)
        # return normalized directions
        return torch.nn.functional.normalize(directions, p=2, dim=-1)

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

    def distill_texture(self) -> None:
        """Distills the texture from the environment map."""
        torch.set_grad_enabled(True)
        texture_width, texture_height = 2048, 1024
        n_iterations_distillation = 1000
        batch_size = 2 ** 21
        texture = torch.nn.Parameter(torch.zeros((1, 4, texture_height, texture_width), dtype=torch.float32, device='cuda'))
        optimizer = torch.optim.AdamW(
            [{'params': texture, 'lr': 1.0}],
            lr=1.0, betas=(0.9, 0.99), eps=1.0e-15,
            weight_decay=0.0, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [LRDecayPolicy(
            lr_init=1.0e-2,
            lr_final=1.0e-3,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=n_iterations_distillation
        )])
        loss_fn = torch.nn.MSELoss()
        for _ in Logger.log_progress(range(n_iterations_distillation), desc='distilling background mlp into texture', miniters=10):
            with torch.no_grad():
                # generate random view directions
                directions = torch.randn((batch_size, 3), dtype=torch.float32, device='cuda')
                directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
                # evaluate the network to get the target values
                directions_tcnn = directions * 0.5 + 0.5  # for compatibility with the SH encoding in tiny-cuda-nn
                target_values = self.net_with_encoding(directions_tcnn).permute(1, 0).float()  # (4, batch_size)
                # convert to texture coordinates
                grid_coords = directions_to_equirectangular_grid_coords(directions)  # (batch_size, 2)

            sampled_values = torch.nn.functional.grid_sample(
                input=texture,  # (1, 4, texture_height, texture_width)
                grid=grid_coords.view(1, batch_size, 1, 2),
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )  # (1, 4, batch_size, 1)
            sampled_values = sampled_values.squeeze()  # (4, batch_size)
            optimizer.zero_grad()
            loss = loss_fn(sampled_values, target_values)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # set distilled texture
        self.register_buffer('distilled_texture', texture.data.contiguous())

    def query_distilled_texture(self, ray_directions: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Returns environment map values for the given camera using the distilled texture."""
        if self.distilled_texture is None:
            self.distill_texture()
        # convert to texture coordinates
        grid_coords = directions_to_equirectangular_grid_coords(ray_directions)  # (height * width, 2)
        # sample from texture
        return torch.nn.functional.grid_sample(
            input=self.distilled_texture,  # (1, 4, texture_height, texture_width)
            grid=grid_coords.view(1, height, width, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze()  # (4, height, width)

    def query_mlp(self, ray_directions: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Returns environment map values for the given ray directions using the MLP."""
        # convert to the format used by the SH encoding in tiny-cuda-nn
        ray_directions = ray_directions * 0.5 + 0.5
        sampled_values = self.net_with_encoding(ray_directions)  # (height * width, 4)
        return sampled_values.permute(1, 0).reshape(4, height, width)

    def forward(self, view: View, use_distilled_texture: bool = False) -> torch.Tensor:
        """Returns environment map values for the given view."""
        ray_directions = self.compute_ray_directions(view)
        if use_distilled_texture:
            return self.query_distilled_texture(ray_directions, view.camera.width, view.camera.height)
        return self.query_mlp(ray_directions, view.camera.width, view.camera.height)
