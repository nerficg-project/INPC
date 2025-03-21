# -- coding: utf-8 --

"""INPC/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Methods.Base.Model import BaseModel
from Optim.Losses.Base import BaseLoss
from Optim.Losses.VGG import VGGLoss
from Optim.Losses.FusedDSSIM import fused_dssim

# @torch.compile  # TORCH_COMPILE_NOTE: uncommenting this makes things slightly faster
def cauchy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Usage initially inspired by Jon Barron's general loss function (https://arxiv.org/abs/1701.03077)."""
    return input.sub(target).mul(5.0).square().mul(0.5).add(1.0).log().mean()


class INPCLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, model: 'BaseModel') -> None:
        super().__init__()
        self.addLossMetric('Cauchy_Color', cauchy_loss, loss_config.LAMBDA_CAUCHY)
        self.addLossMetric('VGG_Color', VGGLoss(), loss_config.LAMBDA_VGG)
        self.addLossMetric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.addLossMetric('Weight_Decay_Hash_Grid', model.appearance_field.normalized_weight_decay, loss_config.LAMBDA_WEIGHT_DECAY)
        if model.tone_mapper is None:
            self.addLossMetric('CRF_Smoothness', lambda: 0.0, 0.0)
        else:
            self.addLossMetric('CRF_Smoothness', model.tone_mapper.response_loss, 1.0)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

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
