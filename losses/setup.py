from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='losses',
    ext_modules=[
        CUDAExtension('losses', [
            'nmdistance.cpp',
            'nmdistance_cuda.cu', ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            include_dirs=["."])
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
