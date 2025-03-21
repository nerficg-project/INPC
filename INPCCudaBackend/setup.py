from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'Provides various CUDA-accelerated functionality for the INPC method.'

ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
sources += glob(str(extension_root / 'utils' / '*.cpp'))
sources += glob(str(extension_root / 'utils' / '*.cu'))
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '*.cpp'))
    sources += glob(str(extension_root / module / 'src' / '*.cu'))

include_dirs = [str(extension_root / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

cxx_flags, nvcc_flags = [], []
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

cuda_extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
)

setup(
    name=extension_name,
    author=__author__,
    packages=[extension_name],
    ext_modules=[cuda_extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
