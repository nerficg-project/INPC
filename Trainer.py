# -- coding: utf-8 --

"""INPC/Trainer.py: Implementation of the trainer for INPC."""

import torch
import matplotlib.pyplot as plt

import Framework
from Logging import Logger
from Datasets.Base import BaseDataset
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback, postTrainingCallback
from Methods.INPC.Loss import INPCLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler
import Thirdparty.TinyCudaNN as tcnn


@Framework.Configurable.configure(
    NUM_ITERATIONS=50_000,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_CAUCHY=1.0,
        LAMBDA_VGG=0.075,
        LAMBDA_DSSIM=0.5,
        LAMBDA_WEIGHT_DECAY=0.1,
    ),
)
class INPCTrainer(GuiTrainer):
    """Defines the trainer for the INPC method."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_sampler = None
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = torch.GradScaler(device='cuda', init_scale=128.0, growth_interval=self.NUM_ITERATIONS + 1)
        self.loss = INPCLoss(self.LOSS, self.model)

    @preTrainingCallback(priority=100)
    @torch.no_grad()
    def setup(self, _, dataset: 'BaseDataset') -> None:
        """Performs all required initializations."""
        dataset.train()
        self.train_sampler = DatasetSampler(dataset=dataset)
        self.model.probability_field.initialize(dataset)
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.setup_exposure_params(dataset)
        param_groups, schedulers = self.model.get_optimizer_param_groups(self.NUM_ITERATIONS)
        try:
            from Thirdparty.Apex import FusedAdam  # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=1.0, betas=(0.9, 0.99), eps=1.0e-15, adam_w_mode=True, weight_decay=0.0)
            Logger.logInfo('using apex FusedAdam')
        except Framework.ExtensionError:
            Logger.logWarning('apex is not installed -> using the slightly slower PyTorch AdamW instead')
            Logger.logWarning('apex can be installed using ./scripts/install.py -e src/Thirdparty/Apex.py')
            self.optimizer = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, 0.99), eps=1.0e-15, weight_decay=0.0, fused=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedulers)

    @trainingCallback(priority=100, start_iteration=500, iteration_stride=100)
    @torch.no_grad()
    def prune(self, *_) -> None:
        """Removes empty leaves from the probability field octree."""
        self.model.probability_field.prune(threshold=0.01)

    @trainingCallback(priority=90, start_iteration=500, iteration_stride=500)
    @torch.no_grad()
    def subdivide(self, *_) -> None:
        """Subdivides the probability field octree."""
        self.model.probability_field.subdivide(threshold=0.5)

    @trainingCallback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds octree leaf count and the learned CRF to default Weights & Biases logging."""
        # log leaf count
        Framework.wandb.log({
            'n_leaves': self.model.probability_field.leaf_centers.shape[0]
        }, step=iteration)
        # update tone mapper if used
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.interpolate_testset_exposures(dataset.test())
        # default logging
        super().logWandB(iteration, dataset)
        # plot current CRF if available
        if self.model.tone_mapper is not None:
            xs = torch.linspace(0.0, 1.0, self.model.tone_mapper.response_params.shape[-1], device='cpu').numpy()
            crf_plot, ax = plt.subplots()
            for ch, color in enumerate(['r', 'g', 'b']):
                ax.plot(xs, self.model.tone_mapper.response_params[ch].clone().detach().flatten().cpu().numpy(), color=color)
            # log CRF plot and leaf count
            Framework.wandb.log({
                'CRF': crf_plot,
            }, step=iteration)

    @trainingCallback(priority=5)
    def training_step(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs an optimization step."""
        dataset.train()
        self.model.train()
        self.loss.train()
        self.optimizer.zero_grad()
        # sample dataset
        camera_properties = self.train_sampler.get(dataset=dataset)['camera_properties']
        dataset.camera.setProperties(camera_properties)
        # render and calculate loss
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = self.renderer.renderImage(camera=dataset.camera)
            loss = self.loss(outputs['rgb'], camera_properties.rgb)
        # update parameters
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()
        if iteration >= 100:
            self.model.probability_field.update(outputs['indices'], outputs['blending_weights'])

    @postTrainingCallback(priority=1000)
    @torch.no_grad()
    def finalize_training(self, _, dataset: 'BaseDataset') -> None:
        """Delete optimization helpers and finalize tone mapper."""
        # set modes
        dataset.test()
        self.model.eval()
        # delete optimization helpers
        self.optimizer.zero_grad()
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = None
        self.loss = None
        self.train_sampler = None
        tcnn.free_temporary_memory()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # finalize tone mapper if used
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.interpolate_testset_exposures(dataset.test())
