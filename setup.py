from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

def extension(name):
    return Extension(
        name, [f"{name}.pyx"],
        include_dirs = [numpy.get_include()],
    )

setup(
    ext_modules=cythonize([
        extension("deconvolve2d"),
        extension("max_pooling_functions"),
        extension("convolve2d"),
    ])
)

# python setup.py build_ext --inplace
