"""INPC/Trainer.py: Implementation of the trainer for INPC."""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

import Framework
from Datasets.utils import apply_background_color
from Logging import Logger
from Datasets.Base import BaseDataset
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.INPC.Loss import INPCLoss
from Methods.INPC.INPCCudaBackend import add_normalized_weight_decay_grads
from Optim.Samplers.DatasetSamplers import DatasetSampler
import Thirdparty.TinyCudaNN as tcnn


@Framework.Configurable.configure(
    NUM_ITERATIONS=50_000,
    PRETRAINED_UNET_WEIGHTS='src/Methods/INPC/unet_pretraining/inpc_pretrained_unet_weights_stage3.pt',
    STORE_PRETRAINING_DATA=False,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_CAUCHY=1.0,
        USE_FUSED_CAUCHY_LOSS=True,
        LAMBDA_VGG=0.075,
        LAMBDA_DSSIM=0.5,
        LAMBDA_WEIGHT_DECAY=0.1,
        USE_FUSED_WEIGHT_DECAY=True,
    ),
)
class INPCTrainer(GuiTrainer):
    """Defines the trainer for the INPC method."""

    def __init__(self, **kwargs) -> None:
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = torch.GradScaler(device='cuda', init_scale=128.0, growth_interval=self.NUM_ITERATIONS + 1)
        if self.LOSS.USE_FUSED_WEIGHT_DECAY:
            # FIXME: should check that ensure the fused version is valid for the requested loss configuration and model
            Logger.log_warning('using fused weight decay -> will not show up in Weights & Biases')
            self.LOSS.LAMBDA_WEIGHT_DECAY = 0.0  # to disable non-fused weight decay loss
        self.loss = INPCLoss(self.LOSS, self.model)

    @pre_training_callback(priority=100)
    @torch.no_grad()
    def setup(self, _, dataset: 'BaseDataset') -> None:
        """Performs all required initializations."""
        if self.PRETRAINED_UNET_WEIGHTS is not None:
            checkpoint_path = Path(self.PRETRAINED_UNET_WEIGHTS)
            if not checkpoint_path.is_file():
                raise Framework.CheckpointError(f'pretrained U-Net weights path "{checkpoint_path}" does not point to a valid file')
            Logger.log_info(f'loading pretrained U-Net weights from "{checkpoint_path}"')
            self.model.unet.set_weights(checkpoint_path)
        dataset.train()
        self.train_sampler = DatasetSampler(dataset=dataset)
        self.model.probability_field.initialize(dataset)
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.setup_exposure_params(dataset)
        param_groups, schedulers = self.model.get_optimizer_param_groups(self.NUM_ITERATIONS)
        try:
            from Thirdparty.Apex import FusedAdam  # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=1.0, betas=(0.9, 0.99), eps=1.0e-15, adam_w_mode=True, weight_decay=0.0)
            Logger.log_info('using apex FusedAdam')
        except Framework.ExtensionError:
            Logger.log_warning('apex is not installed -> using the slightly slower PyTorch AdamW instead')
            Logger.log_warning('apex can be installed using ./scripts/install.py -e src/Thirdparty/Apex.py')
            self.optimizer = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, 0.99), eps=1.0e-15, weight_decay=0.0, fused=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedulers)
        torch.cuda.empty_cache()

    @training_callback(priority=100, start_iteration=500, iteration_stride=100)
    @torch.no_grad()
    def prune(self, *_) -> None:
        """Removes empty leaves from the probability field octree."""
        self.model.probability_field.prune(threshold=0.01)
        torch.cuda.empty_cache()

    @training_callback(priority=90, start_iteration=500, iteration_stride=500)
    @torch.no_grad()
    def subdivide(self, *_) -> None:
        """Subdivides the probability field octree."""
        self.model.probability_field.subdivide(threshold=0.5)
        torch.cuda.empty_cache()

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds octree leaf count and the learned CRF to default Weights & Biases logging."""
        # disable inference optimizations for wandb logging
        use_compiled_unet_inference = self.renderer.USE_COMPILED_UNET_INFERENCE
        use_distilled_bg_texture_inference = self.renderer.USE_DISTILLED_BG_TEXTURE_INFERENCE
        self.renderer.USE_COMPILED_UNET_INFERENCE = False
        self.renderer.USE_DISTILLED_BG_TEXTURE_INFERENCE = False
        # log leaf count
        Framework.wandb.log({
            'n_leaves': self.model.probability_field.leaf_centers.shape[0]
        }, step=iteration)
        # update tone mapper if used
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.interpolate_testset_exposures(dataset.test())
        # default logging
        super().log_wandb(iteration, dataset)
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
        # restore inference optimizations
        self.renderer.USE_COMPILED_UNET_INFERENCE = use_compiled_unet_inference
        self.renderer.USE_DISTILLED_BG_TEXTURE_INFERENCE = use_distilled_bg_texture_inference

    @training_callback(active=False, priority=6, iteration_stride=10)
    def render_convergence_anim(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Renders a test view for convergence animation."""
        output_directory = self.output_directory / 'convergence_anim'
        output_directory.mkdir(parents=True, exist_ok=True)
        from Datasets.utils import save_image
        dataset.test()
        self.model.eval()
        old_n_multisamples = self.renderer.N_MULTISAMPLES
        old_use_expected_samples = self.renderer.USE_EXPECTED_SAMPLES
        self.renderer.N_MULTISAMPLES = 1
        self.renderer.USE_EXPECTED_SAMPLES = False
        # update tone mapper if used
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.interpolate_testset_exposures(dataset.test())
        rgb = self.renderer.render_image(view=dataset[1], to_chw=True)['rgb']
        save_image(output_directory / f'{iteration:05d}.png', rgb)
        self.renderer.N_MULTISAMPLES = old_n_multisamples
        self.renderer.USE_EXPECTED_SAMPLES = old_use_expected_samples

    @training_callback(priority=5)
    def training_step(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs an optimization step."""
        dataset.train()
        self.model.train()
        self.loss.train()
        self.optimizer.zero_grad()
        # sample viewpoint
        view = self.train_sampler.get(dataset=dataset)['view']
        # render and calculate loss
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = self.renderer.render_image(view=view)
            # compose gt with background color if needed  # FIXME: integrate into data model
            rgb_gt = view.rgb
            if (alpha_gt := view.alpha) is not None:
                rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
            loss = self.loss(outputs['rgb'], rgb_gt)
        # update parameters
        self.grad_scaler.scale(loss).backward()
        if self.LOSS.USE_FUSED_WEIGHT_DECAY:
            self.grad_scaler.unscale_(self.optimizer)
            add_normalized_weight_decay_grads(self.model.appearance_field.hash_grid.params)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()
        if iteration >= 100:
            self.model.probability_field.update(outputs['indices'], outputs['blending_weights'])
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.apply_response_constraints()

    @post_training_callback(priority=1000)
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
        # distill background texture
        if self.model.background_model is not None:
            self.model.background_model.distill_texture()
        # finalize tone mapper if used
        if self.model.tone_mapper is not None:
            self.model.tone_mapper.interpolate_testset_exposures(dataset.test())

    @post_training_callback(priority=900)
    @torch.no_grad()
    def save_unet_weights(self, *_):
        """Saves the U-Net weights in a separate file."""
        self.model.unet.save_weights(self.checkpoint_directory / 'unet.pt')

    @post_training_callback(active='STORE_PRETRAINING_DATA', priority=800)
    @torch.no_grad()
    def store_pretraining_data(self, _, dataset: 'BaseDataset') -> None:
        """Writes the U-Net input and target output for all training views for later usage during U-Net pretraining."""
        dataset.train()
        self.model.eval()
        output_directory = self.output_directory / 'pretraining_data'
        output_directory.mkdir(parents=True, exist_ok=True)
        Logger.log_info(f'extracting U-Net pretraining data from {dataset.mode} set images')
        old_n_multisamples = self.renderer.N_MULTISAMPLES
        old_use_expected_samples = self.renderer.USE_EXPECTED_SAMPLES
        self.renderer.N_MULTISAMPLES = 1
        self.renderer.USE_EXPECTED_SAMPLES = False
        for index, view in Logger.log_progress(enumerate(dataset), total=len(dataset), desc='view', leave=False):
            input_image = self.renderer.render_image(view=view, to_chw=True)['feature_image']
            # compose gt with background color if needed  # FIXME: integrate into data model
            rgb_gt = view.rgb
            if (alpha_gt := view.alpha) is not None:
                rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
            target_image = rgb_gt.double()  # use double precision for better accuracy
            if self.model.tone_mapper is not None:
                target_image = self.model.tone_mapper.inverse_forward(target_image, view)
            target_image = target_image.float()
            combined_data = torch.cat((input_image, target_image), dim=0).cpu()  # (7, H, W)
            torch.save(combined_data, output_directory / f'{index:05d}.pt')
        self.renderer.N_MULTISAMPLES = old_n_multisamples
        self.renderer.USE_EXPECTED_SAMPLES = old_use_expected_samples
