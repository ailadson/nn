from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        "deconvolve2d.pyx",
        "max_pooling_functions.pyx",
        "convolve2d.pyx",
    ]), include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace
