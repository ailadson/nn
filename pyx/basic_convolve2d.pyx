# This code is a basic Cython implementation of convolution. It has
# been surpassed by the avx_convolve2d code.

cimport cython
cimport numpy as np

ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void convolve_step(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] target,
    DTYPE_t kval,
    int kernel_offset_i,
    int kernel_offset_j) nogil:

    cdef int num_rows = ipt.shape[0]
    cdef int num_cols = ipt.shape[1]
    cdef int i, j
    cdef int i2, j2

    for i in range(num_rows):
        for j in range(num_cols):
            i2 = i + kernel_offset_i
            j2 = j + kernel_offset_j

            if (i2 < 0) | (i2 > num_rows):
                continue
            if (j2 < 0) | (j2 > num_rows):
                continue

            target[i, j] += ipt[i2, j2] * kval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void convolve2d_(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target) nogil:

    cdef int krows = kernel.shape[0]
    cdef int kcols = kernel.shape[1]
    cdef int mid_i = kernel.shape[0] // 2
    cdef int mid_j = kernel.shape[1] // 2
    cdef int i, j

    for i in range(krows):
        for j in range(kcols):
            convolve_step(
                ipt, target, kernel[i, j], i - mid_i, j - mid_j
            )

@cython.boundscheck(False)
cdef void backward_convolve2d_(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target) nogil:

    cdef int krows = kernel.shape[0]
    cdef int kcols = kernel.shape[1]
    # If the kernel has a non-odd KSIZE I believe the center is
    # adjusted slightly... This code may be wrong though.
    cdef int mid_i = (krows - 1) - (krows // 2)
    cdef int mid_j = (kcols - 1) - (kcols // 2)
    cdef int i, j

    for i in range(krows):
        for j in range(kcols):
            # Backward convolution means flipping the kernel
            # left/right and up/down.
            i = (krows - 1) - i
            j = (kcols - 1) - j
            convolve_step(
                ipt, target, kernel[i, j], i - mid_i, j - mid_j
            )

def convolve2d(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target):

    convolve2d_(ipt, kernel, target)
