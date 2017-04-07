cimport numpy as np

cdef extern from "matrix.h":
    ctypedef struct shape_t:
        size_t height
        size_t width

cdef void avx_convolve2d_(
    np.float32_t[:, :] ipt,
    np.float32_t[:, :] kernel,
    np.float32_t[:, :] destination) nogil
