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
        Extension("main", ["main.pyx"])
    ])
)

# python setup.py build_ext --inplace
