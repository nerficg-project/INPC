from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .INPCCudaBackend.torch_bindings.rasterization import INPCRasterizer, RasterizerSettings, RasterizerMode
    from .INPCCudaBackend.torch_bindings.sampling import ProbabilityFieldSampler, compute_viewpoint_weights
    from .INPCCudaBackend.torch_bindings.misc import compute_normalized_weight_decay_grads, add_normalized_weight_decay_grads, spherical_contraction, fused_cauchy_loss
    __all__ = [
        'INPCRasterizer', 'RasterizerSettings', 'RasterizerMode',
        'ProbabilityFieldSampler', 'compute_viewpoint_weights',
        'compute_normalized_weight_decay_grads', 'add_normalized_weight_decay_grads', 'spherical_contraction', 'fused_cauchy_loss'
    ]
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
