# -- coding: utf-8 --

"""
INPC/Modules/AppearanceField.py: Appearance field module.
"""

import torch

import Framework
from Methods.INPC.utils import LRDecayPolicy, spherical_contraction, feature_to_opacity
import Thirdparty.TinyCudaNN as tcnn


class AppearanceField(torch.nn.Module):
    """Positional encoding based on the hash grid used in Instant-NGP (https://arxiv.org/abs/2201.05989)."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.hash_grid = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,  # 3d position
            n_output_dims=37,  # SH degree 2 for 4 channels -> 36 features + 1 for opacity
            encoding_config={
                'otype': 'Grid',
                'type': 'Hash',
                'n_levels': 10,
                'n_features_per_level': 4,
                'log2_hashmap_size': 23,
                'base_resolution': 16,
                'per_level_scale': 2.0,
                'interpolation': 'Linear',
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 64,
                'n_hidden_layers': 1,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.hash_grid.parameters(), 'lr': 1.0}]
        schedulers = [LRDecayPolicy(
            lr_init=1.0e-2,
            lr_final=1.0e-2 / 30,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=max_iterations
        )]
        return param_groups, schedulers

    # @torch.compile  # TORCH_COMPILE_NOTE: uncommenting this makes things slightly faster
    def normalized_weight_decay(self) -> torch.Tensor:
        """Normalized weight decay as proposed in Zip-NeRF (https://arxiv.org/abs/2304.06706)."""
        # FIXME: using torch.compile results in 20% faster training but can cause a quality reduction
        #  -> likely due to the division being applied element-wise instead of on the sums
        return (
            self.hash_grid.params[6_144:22_528].square().mean() +
            self.hash_grid.params[22_528:153_600].square().mean() +
            self.hash_grid.params[153_600:1_202_176].square().mean() +
            self.hash_grid.params[1_202_176:9_590_784].square().mean() +
            self.hash_grid.params[9_590_784:43_145_216].square().mean() +
            self.hash_grid.params[43_145_216:76_699_648].square().mean() +
            self.hash_grid.params[76_699_648:110_254_080].square().mean() +
            self.hash_grid.params[110_254_080:143_808_512].square().mean() +
            self.hash_grid.params[143_808_512:177_362_944].square().mean() +
            self.hash_grid.params[177_362_944:210_917_376].square().mean()
        )

    def forward(self, positions: torch.Tensor, batch_size: int, return_raw: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Computes the appearance of the given positions."""
        positions = spherical_contraction(positions)
        n_points = positions.shape[0]
        if n_points > batch_size:
            features_raw = torch.empty(n_points, 37, device=positions.device, dtype=torch.half)
            for i in range(0, n_points, batch_size):
                j = min(i + batch_size, n_points)
                features_raw[i:j] = self.hash_grid(positions[i:j])
        else:
            features_raw = self.hash_grid(positions)
        if return_raw:
            return features_raw.contiguous()
        opacities, features = features_raw.split([1, 36], dim=-1)
        opacities = feature_to_opacity(opacities)
        return opacities.contiguous(), features.contiguous()
