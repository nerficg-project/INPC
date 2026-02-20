"""INPC/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Methods.Base.Model import BaseModel
from Optim.Losses.Base import BaseLoss
from Optim.Losses.VGG import VGGLoss
from Optim.Losses.DSSIM import fused_dssim
from Methods.INPC.INPCCudaBackend import fused_cauchy_loss


def cauchy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Usage initially inspired by Jon Barron's general loss function (https://arxiv.org/abs/1701.03077)."""
    return input.sub(target).square().mul(12.5).log1p().mean()


class INPCLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, model: 'BaseModel') -> None:
        super().__init__()
        self.add_loss_metric('Cauchy_Color', fused_cauchy_loss if loss_config.USE_FUSED_CAUCHY_LOSS else cauchy_loss, loss_config.LAMBDA_CAUCHY)
        self.add_loss_metric('VGG_Color', VGGLoss(), loss_config.LAMBDA_VGG)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('Weight_Decay_Hash_Grid', model.appearance_field.normalized_weight_decay, loss_config.LAMBDA_WEIGHT_DECAY)
        if model.tone_mapper is None:
            self.add_loss_metric('CRF_Smoothness', lambda: 0.0, 0.0)
        else:
            self.add_loss_metric('CRF_Smoothness', model.tone_mapper.response_loss, 1.0)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    @torch.autocast('cuda', dtype=torch.float32)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward({
            'Cauchy_Color': {'input': input, 'target': target},
            'VGG_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'Weight_Decay_Hash_Grid': {},
            'CRF_Smoothness': {},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })


class UNetPretrainingLoss:
    def __init__(self) -> None:
        self.vgg_loss = VGGLoss()

    @torch.autocast('cuda', dtype=torch.float32)
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fused_cauchy_loss(input, target) + 0.5 * fused_dssim(input, target) + 0.075 * self.vgg_loss(input, target)
