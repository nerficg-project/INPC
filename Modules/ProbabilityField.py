# -- coding: utf-8 --

"""
INPC/Modules/ProbabilityField.py: Probability field module.
"""

import torch
import math

import Framework
from Logging import Logger
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import tensor_to_string
from Methods.INPC.utils import compute_halton_coords
from Methods.INPC.INPCCudaBackend import ProbabilityFieldSampler, compute_viewpoint_weights
from CudaUtils.MortonEncoding import morton_encode


class ProbabilityField(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.cuda_backend = ProbabilityFieldSampler(Framework.config.GLOBAL.RANDOM_SEED)

    def get_current_weights(self, camera: PerspectiveCamera | None) -> torch.Tensor:
        """Returns the current weights of the octree leaves."""
        if camera is not None:
            weights = compute_viewpoint_weights(
                centers=self.leaf_centers,
                levels=self.leaf_levels,
                weights=self.leaf_weights,
                camera=camera,
                initial_size=self.initial_cell_size[0].item()
            )
        else:
            weights = self.leaf_weights.clone()
            weights *= torch.exp2(-self.leaf_levels * 0.5)
        return weights

    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def generate_samples(self, n_samples: int, camera: PerspectiveCamera | None = None, ensure_visibility: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate sample position from the octree."""
        # compute weights
        weights = self.get_current_weights(camera)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = self.leaf_centers[indices]
        # add random offset
        samples.add_(torch.rand_like(samples).sub_(0.5).mul_(self.initial_cell_size / 2 ** self.leaf_levels[indices, None]))
        # ensure samples are visible during training # TODO: a C++/CUDA implementation could probably save some time here
        if camera is not None and ensure_visibility:
            visibility_mask = camera.projectPoints(samples)[1]
            samples = samples[visibility_mask]
            indices = indices[visibility_mask]
            # re-sample until enough samples are visible, try to avoid more than one re-sample by doing some math
            expected_visibility_ratio = n_samples / samples.shape[0]
            undersampling_protection = 10_000
            while samples.shape[0] < n_samples:
                n_missing = n_samples - samples.shape[0]
                n_extra_samples = math.ceil(n_missing * expected_visibility_ratio) + undersampling_protection
                n_extra_samples = min(100_000_000, n_extra_samples)  # limit to avoid OOM for extreme cases
                extra_samples, extra_indices = self.rejection_sample(n_extra_samples, weights, camera)
                n_used = min(n_samples - samples.shape[0], extra_samples.shape[0])
                extra_samples = extra_samples[:n_used]
                extra_indices = extra_indices[:n_used]
                # we cat inside the loop as we expect a single iteration  # TODO: could pre-allocate to avoid cat
                samples = torch.cat((samples, extra_samples), dim=0)
                indices = torch.cat((indices, extra_indices), dim=0)
        return samples, indices

    def rejection_sample(self, n_samples: int, weights: torch.Tensor, camera: PerspectiveCamera) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-sample positions with given weights. Used to avoid recursion when including cell corners."""
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = self.leaf_centers[indices]
        # add random offset
        samples.add_(torch.rand_like(samples).sub_(0.5).mul_(self.initial_cell_size / 2 ** self.leaf_levels[indices, None]))
        # ensure samples are visible
        visibility_mask = camera.projectPoints(samples)[1]
        samples = samples[visibility_mask]
        indices = indices[visibility_mask]
        return samples, indices

    def generate_multisamples(self, n_samples: int, n_multi: int = 4, camera: PerspectiveCamera | None = None) -> torch.Tensor:
        """Generate multiple sets of sample positions."""
        return self.cuda_backend.generate_samples(
            centers=self.leaf_centers,
            levels=self.leaf_levels,
            weights=self.leaf_weights,
            camera=camera,
            n_samples=n_samples * n_multi,
            initial_size=self.initial_cell_size[0].item()
        )

    def generate_expected_samples(self, n_samples: int, n_multi: int, camera: PerspectiveCamera) -> torch.Tensor:
        """Generate multiple sets of positions samples."""
        return self.cuda_backend.generate_expected_samples(
            centers=self.leaf_centers,
            levels=self.leaf_levels,
            weights=self.leaf_weights,
            camera=camera,
            n_samples=n_samples,
            n_multi=n_multi,
            initial_size=self.initial_cell_size[0].item()
        )

    @torch.autocast('cuda', enabled=False)
    def extract_global(self, n_samples: int, use_morton_order: bool) -> torch.Tensor:
        """Generate quasi-random sample positions from the octree."""
        weights = self.get_current_weights(None)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        counts = torch.bincount(indices, minlength=weights.shape[0])
        cell_indices_samples = torch.arange(weights.shape[0], dtype=torch.int32, device=weights.device).repeat_interleave(counts)
        halton_coords = compute_halton_coords(counts, cell_indices_samples)
        cell_size = self.initial_cell_size / 2 ** self.leaf_levels[cell_indices_samples, None]
        initial_samples = self.leaf_centers[cell_indices_samples]
        samples = initial_samples - (cell_size / 2) + halton_coords * cell_size
        # move samples back to cell center if cell has only one sample
        single_sample_mask = (counts == 1).repeat_interleave(counts)
        samples[single_sample_mask] = initial_samples[single_sample_mask]
        if use_morton_order:
            morton_encoding = morton_encode(samples)
            samples = samples[torch.argsort(morton_encoding)].contiguous()
        return samples

    @torch.no_grad()
    def update(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        """Update the leaves with the given indices and weights."""
        self.leaf_weights.mul_(0.9968)
        self.leaf_subdivision_scores.mul_(0.9968)
        weights = weights.squeeze().to(self.leaf_weights)
        max_sampled = torch.zeros_like(self.leaf_weights)
        min_sampled = torch.zeros_like(self.leaf_weights)
        max_sampled.scatter_reduce_(dim=0, index=indices, src=weights, reduce='amax', include_self=False)
        min_sampled.scatter_reduce_(dim=0, index=indices, src=weights, reduce='amin', include_self=False)
        torch.maximum(self.leaf_weights, max_sampled, out=self.leaf_weights)
        torch.maximum(self.leaf_subdivision_scores, max_sampled.sub_(min_sampled).clamp_(0.0, 1.0), out=self.leaf_subdivision_scores)

    def prune(self, threshold: float) -> None:
        """Prune leaves with occupancy < threshold."""
        mask = self.leaf_weights > threshold
        self.leaf_centers.data = self.leaf_centers[mask].contiguous()
        self.leaf_levels.data = self.leaf_levels[mask].contiguous()
        self.leaf_weights.data = self.leaf_weights[mask].contiguous()
        self.leaf_subdivision_scores.data = self.leaf_subdivision_scores[mask].contiguous()

    def initialize(self, dataset: BaseDataset) -> None:
        """Initialize the octree based on the given dataset."""
        # create initial octree leaves
        bb_min, bb_max = dataset.getBoundingBox().cuda()
        axis_lengths = (bb_max - bb_min).abs()
        scaling_factor = (2_097_152 / torch.prod(axis_lengths)) ** (1 / 3)
        resolution = torch.round(axis_lengths * scaling_factor).int()
        initial_cell_size = axis_lengths / resolution
        # ensure cells are cubic
        initial_cell_size = initial_cell_size.max()
        new_size = initial_cell_size * resolution
        offset = (new_size - axis_lengths) / 2
        bb_min -= offset
        bb_max += offset
        Logger.logDebug(f'enlarged bounding box by {tensor_to_string(offset, precision=4)} to ensure cubic cells')
        half_cell_size = initial_cell_size / 2
        start = bb_min + half_cell_size
        end = bb_max - half_cell_size
        centers = torch.stack(torch.meshgrid(
            torch.linspace(start[0], end[0], resolution[0].item(), dtype=torch.float32, device='cuda'),
            torch.linspace(start[1], end[1], resolution[1].item(), dtype=torch.float32, device='cuda'),
            torch.linspace(start[2], end[2], resolution[2].item(), dtype=torch.float32, device='cuda'),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)
        # initialize leaf weights according to sfm points
        if dataset.point_cloud is not None:
            positions = dataset.point_cloud.positions.clone().cuda()
            positions = (positions - bb_min) / (bb_max - bb_min)
            indices_3d = torch.round(positions * (resolution - 1)).long()
            indices = indices_3d[:, 2] + resolution[2] * indices_3d[:, 1] + resolution[2] * resolution[1] * indices_3d[:, 0]
            indices, counts = indices.unique(dim=0, return_counts=True)
            counts = counts.float()
            # normalize counts to [0.1, 1.0] based on quantile and clamp
            counts = (counts / torch.quantile(counts, 0.95)).clamp_(0.1, 1.0)
            initial_weights = torch.full((centers.shape[0],), fill_value=0.1, dtype=torch.float32, device='cuda')
            initial_weights[indices] = counts
        else:
            initial_weights = torch.full((centers.shape[0],), fill_value=1.0, dtype=torch.float32, device='cuda')
        # setup buffers
        self.register_buffer('initial_cell_size', initial_cell_size.expand(3))
        self.register_buffer('leaf_centers', centers.contiguous())
        self.register_buffer('leaf_levels', torch.zeros((centers.shape[0],), dtype=torch.int32, device='cuda'))
        self.register_buffer('leaf_weights', initial_weights.contiguous())
        self.register_buffer('leaf_subdivision_scores', torch.zeros_like(self.leaf_weights))
        # carve octree based on training viewpoints
        self.carve(dataset.train())

    def subdivide(self, threshold: float) -> None:
        """Subdivide leaves with subdivision score > threshold."""
        # determine which leaves to subdivide
        valid_mask = (self.leaf_subdivision_scores > threshold) & (self.leaf_levels < 6)
        # check if octree would exceed 256^3 leaves after subdivision
        n_valid_leaves = valid_mask.sum()
        if n_valid_leaves * 8 + (self.leaf_centers.shape[0] - n_valid_leaves) > 16_777_216:
            return
        valid_centers = self.leaf_centers[valid_mask]
        valid_levels = self.leaf_levels[valid_mask]
        valid_weights = self.leaf_weights[valid_mask]
        # create new leaves
        new_centers = torch.repeat_interleave(valid_centers, 8, dim=0)
        new_levels = torch.repeat_interleave(valid_levels + 1, 8, dim=0)
        new_weights = torch.repeat_interleave(valid_weights, 8, dim=0)
        new_subdivision_scores = torch.zeros_like(new_weights)
        # calculate offset
        offset = self.initial_cell_size / (2 ** (new_levels + 1))[:, None]  # (n_new_leaves, 3)
        # TODO: optimized implementation (see get_leaf_corners)
        # x position
        new_centers[0::2, 0] -= offset[0::2, 0]
        new_centers[1::2, 0] += offset[1::2, 0]
        # y position
        new_centers[0::4, 1] -= offset[0::4, 1]
        new_centers[1::4, 1] -= offset[1::4, 1]
        new_centers[2::4, 1] += offset[2::4, 1]
        new_centers[3::4, 1] += offset[3::4, 1]
        # z position
        new_centers[0::8, 2] -= offset[0::8, 2]
        new_centers[1::8, 2] -= offset[1::8, 2]
        new_centers[2::8, 2] -= offset[2::8, 2]
        new_centers[3::8, 2] -= offset[3::8, 2]
        new_centers[4::8, 2] += offset[4::8, 2]
        new_centers[5::8, 2] += offset[5::8, 2]
        new_centers[6::8, 2] += offset[6::8, 2]
        new_centers[7::8, 2] += offset[7::8, 2]
        # append new leaves
        self.leaf_centers.data = torch.cat([self.leaf_centers[~valid_mask], new_centers], dim=0).contiguous()
        self.leaf_levels.data = torch.cat([self.leaf_levels[~valid_mask], new_levels], dim=0).contiguous()
        self.leaf_weights.data = torch.cat([self.leaf_weights[~valid_mask], new_weights], dim=0).contiguous()
        self.leaf_subdivision_scores.data = torch.cat([self.leaf_subdivision_scores[~valid_mask], new_subdivision_scores], dim=0).contiguous()

    def carve(self, dataset: BaseDataset) -> None:
        """Carve the octree based on the dataset."""
        leaf_corners = self._get_leaf_corners().reshape(-1, 3)  # (n_leaves * 8, 3)
        remaining_cells = torch.zeros((self.leaf_centers.shape[0],), dtype=torch.bool, device=self.leaf_centers.device)
        for camera_properties in Logger.logProgressBar(dataset, desc='training viewpoint carving', leave=False):
            dataset.camera.setProperties(camera_properties)
            valid_mask = dataset.camera.projectPoints(leaf_corners)[1]
            remaining_cells |= valid_mask.reshape(-1, 8).any(dim=1)
        self.leaf_centers.data = self.leaf_centers[remaining_cells].contiguous()
        self.leaf_levels.data = self.leaf_levels[remaining_cells].contiguous()
        self.leaf_weights.data = self.leaf_weights[remaining_cells].contiguous()
        self.leaf_subdivision_scores.data = self.leaf_subdivision_scores[remaining_cells].contiguous()
        Logger.logDebug(f'removed {(~remaining_cells).sum().item():,} leaves that are not visible from any viewpoint')

    def _get_leaf_corners(self) -> torch.Tensor:
        """Returns the corners of the octree leaves."""
        # calculate half leaf size
        half_leaf_size = self.initial_cell_size / (2 ** (self.leaf_levels + 1))[:, None]  # (n_leaves, 3)
        leaf_corners = torch.repeat_interleave(self.leaf_centers, 8, dim=0).reshape(-1, 8, 3)  # (n_leaves, 8, 3)
        offset_directions = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [+1.0, +1.0, +1.0],
        ], dtype=leaf_corners.dtype, device=leaf_corners.device)  # (8, 3)
        offsets = offset_directions.mul(half_leaf_size[:, None, :])  # (n_leaves, 8, 3)
        leaf_corners.add_(offsets)
        return leaf_corners
