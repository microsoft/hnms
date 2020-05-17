import os.path as op
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_folder = 'hnms/extension/'

include_dirs = [op.join(ext_folder, 'include')]

setup(
    name='hnms',
    ext_modules=[
        CUDAExtension('hnms._c', [
            op.join(ext_folder, 'src/hnms_module.cpp'),
            op.join(ext_folder, 'src/cuda/hnms.cu'),
            op.join(ext_folder, 'src/cpu/hnms.cpp'),
        ], include_dirs=include_dirs),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
