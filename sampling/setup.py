from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sampling',
    ext_modules=[
        CUDAExtension('sampling', [
            'sampling.cpp',
            'sampling_cuda.cu',],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
