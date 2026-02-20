from typing import Any
import torch
from torch.autograd.function import once_differentiable

import Framework
from INPCCudaBackend import _C


def compute_normalized_weight_decay_grads(weights: torch.Tensor) -> torch.Tensor:
    return _C.compute_normalized_weight_decay_grads_cuda(weights)


def add_normalized_weight_decay_grads(param: torch.nn.Parameter) -> torch.Tensor:
    return _C.add_normalized_weight_decay_grads_cuda(param.data, param.grad)


def spherical_contraction(positions: torch.Tensor) -> torch.Tensor:
    return _C.spherical_contraction_cuda(positions)


class _CauchyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = _C.cauchy_loss_cuda(input, target)
        ctx.save_for_backward(input, target)
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_input = _C.cauchy_loss_backward_cuda(grad, *ctx.saved_tensors)
        return grad_input, None


def fused_cauchy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the Cauchy loss between input and target tensors and do mean reduction."""
    if input.shape != target.shape:
        raise Framework.LossError(f'input and target shapes must match ({input.shape} vs. {target.shape})')
    return _CauchyLoss.apply(input, target)
