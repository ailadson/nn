cimport cython
import numpy as np
cimport numpy as np

cdef struct bounds_s:
    int start_row
    int end_row
    int start_col
    int end_col

cdef int max(int a, int b) nogil:
    if a > b:
        return a
    return b

cdef int min(int a, int b) nogil:
    if a > b:
        return b
    return a

cdef bounds_s get_deconvolve_bounds(
    np.float64_t[:, :] mat, int offset_row, int offset_col) nogil:

    cdef int num_of_rows = mat.shape[0]
    cdef int num_of_cols = mat.shape[1]
    cdef bounds_s bounds

    bounds.start_row = max(offset_row, 0)
    bounds.end_row = min(offset_row + num_of_rows, num_of_rows)
    bounds.start_col = max(offset_col, 0)
    bounds.end_col = min(offset_col + num_of_cols, num_of_cols)

    return bounds

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t sum_el_prod2d(
      np.float64_t[:, :] ipt,
      np.float64_t[:, :] err,
      int ipt_start_row,
      int ipt_start_col,
      int err_start_row,
      int err_start_col,
      int num_rows,
      int num_cols) nogil:

    cdef np.float64_t s = 0
    cdef int i, j
    cdef np.float64_t ipt_el, err_el

    for i in range(num_rows):
        for j in range(num_cols):
            ipt_el = ipt[ipt_start_row + i, ipt_start_col + j]
            err_el = err[err_start_row + i, err_start_col + j]
            s += ipt_el * err_el
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void deconvolve2d_(
        np.float64_t[:, :] ipt,
        np.float64_t[:, :] error,
        np.float64_t[:, :] deriv_filter) nogil:

    cdef int num_of_rows = deriv_filter.shape[0]
    cdef int num_of_cols = deriv_filter.shape[1]
    cdef int i, j
    cdef int offset_row, offset_col
    cdef bounds_s ipt_bounds
    cdef bounds_s error_bounds

    for i in range(num_of_rows):
        for j in range(num_of_cols):
            offset_row = i - (num_of_rows // 2)
            offset_col = j - (num_of_cols // 2)
            ipt_bounds = get_deconvolve_bounds(
                ipt, offset_row, offset_col
            )
            error_bounds = get_deconvolve_bounds(
                error, -1 * offset_row, -1 * offset_col
            )

            # += meant for accumulating errors in the deriv filter
            deriv_filter[i, j] += sum_el_prod2d(
                ipt, error,
                ipt_bounds.start_row, ipt_bounds.start_col,
                error_bounds.start_row, error_bounds.start_col,
                ipt_bounds.end_row - ipt_bounds.start_row,
                ipt_bounds.end_col - ipt_bounds.start_col
            )

def deconvolve2d(
        np.float64_t[:, :] ipt,
        np.float64_t[:, :] error,
        np.float64_t[:, :] deriv_filter):

    with nogil:
        deconvolve2d_(ipt, error, deriv_filter)
