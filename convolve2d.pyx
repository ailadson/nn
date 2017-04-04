cimport cython
cimport numpy as np

@cython.boundscheck(False)
cdef void convolve_step(
    np.float64_t[:, :] ipt,
    np.float64_t[:, :] target,
    np.float64_t kval,
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
cdef void convolve2d_(
    np.float64_t[:, :] ipt,
    np.float64_t[:, :] kernel,
    np.float64_t[:, :] target) nogil:

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

def convolve2d(
    np.float64_t[:, :] ipt,
    np.float64_t[:, :] kernel,
    np.float64_t[:, :] target):

    with nogil:
        convolve2d_(ipt, kernel, target)
