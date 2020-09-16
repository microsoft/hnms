import os.path as op
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


ext_folder = 'hnms/extension/'

include_dirs = [op.join(ext_folder, 'include')]

include_dirs = [op.abspath(i) for i in include_dirs]

define_macros = []

extension = CppExtension

if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
    extension = CUDAExtension
    define_macros += [("WITH_CUDA", None)]


setup(
    name='hnms',
    ext_modules=[
        extension(
            'hnms._c',
            [
                op.join(ext_folder, 'src/hnms_module.cpp'),
                op.join(ext_folder, 'src/cuda/hnms.cu'),
                op.join(ext_folder, 'src/cpu/hnms.cpp'),
            ],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
