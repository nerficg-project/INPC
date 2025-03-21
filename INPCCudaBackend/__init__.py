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
    from .INPCCudaBackend.torch_bindings.rasterization import INPCRasterizer
    from .INPCCudaBackend.torch_bindings.sampling import ProbabilityFieldSampler, compute_viewpoint_weights
    __all__ = ['INPCRasterizer', 'ProbabilityFieldSampler', 'compute_viewpoint_weights']
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
