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

        Extension("avx_convolve2d_main_py",
                  ["./avx_convolve2d_py/main.pyx"],
                  include_dirs = ["./avx_convolve2d_py", "./"])
    ])
)

# python setup.py build_ext --inplace
