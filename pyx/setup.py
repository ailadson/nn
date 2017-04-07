from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

def extension(name):
    return Extension(
        f"{name}", [f"{name}.pyx"],
        include_dirs = [numpy.get_include(), "../c", "./"],
    )

setup(
    ext_modules=cythonize([
        extension("deconvolve2d"),
        extension("max_pooling_functions"),
        extension("convolve2d"),
        extension("avx_convolve2d_py"),
    ])
)
