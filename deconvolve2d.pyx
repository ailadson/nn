import numpy as np
cimport numpy as np

# ctypedef np.ndarray[np.float64_t, ndim=2] np.ndarray[np.float64_t, ndim=2]
# ctypedef np.float64_t[:] np.ndarray[np.float64_t, ndim=2]

def max(int a, int b):
    if a > b:
        return a
    return b

def min(int a, int b):
    if a > b:
        return b
    return a

def get_deconvolve_bounds(np.ndarray[np.float64_t, ndim=2] mat, int offset_row, int offset_col):
    cdef int num_of_rows = mat.shape[0]
    cdef int num_of_cols = mat.shape[1]
    cdef int start_row = max(offset_row, 0)
    cdef int end_row = min(offset_row + num_of_rows, num_of_rows)
    cdef int start_col = max(offset_col, 0)
    cdef int end_col = min(offset_col + num_of_cols, num_of_cols)
    return (start_row, end_row, start_col, end_col)

def sum_el_prod2d(
      np.ndarray[np.float64_t, ndim=2] ipt,
      np.ndarray[np.float64_t, ndim=2] err,
      int ipt_start_row,
      int ipt_start_col,
      int err_start_row,
      int err_start_col,
      int num_rows,
      int num_cols):

    cdef np.float64_t s = 0
    cdef int i, j
    cdef np.float64_t ipt_el, err_el

    for i in range(num_rows):
        for j in range(num_cols):
            ipt_el = ipt[ipt_start_row + i, ipt_start_col + j]
            err_el = err[err_start_row + i, err_start_col + j]
            s += ipt_el * err_el
    return s



def deconvolve2d(np.ndarray[np.float64_t, ndim=2] ipt, np.ndarray[np.float64_t, ndim=2] error, np.ndarray[np.float64_t, ndim=2] deriv_filter):
    cdef int num_of_rows = len(deriv_filter)
    cdef int num_of_cols = len(deriv_filter[0])
    cdef int i, j
    cdef int offset_row, offset_col
    cdef int ipt_start_row, ipt_end_row, ipt_start_col, ipt_end_col
    cdef int error_start_row, error_end_row, error_start_col, error_end_col

    for i in range(num_of_rows):
        for j in range(num_of_cols):
            offset_row = i - num_of_rows // 2
            offset_col = j - num_of_cols // 2
            ipt_start_row, ipt_end_row, ipt_start_col, ipt_end_col = (
                get_deconvolve_bounds(ipt, offset_row, offset_col)
            )
            error_start_row, error_end_row, error_start_col, error_end_col = (
                get_deconvolve_bounds(error, -1 * offset_row, -1 * offset_col)
            )

            # += meant for accumulating errors in the deriv filter
            deriv_filter[i, j] += sum_el_prod2d(
                ipt, error,
                ipt_start_row, ipt_start_col,
                error_start_row, error_start_col,
                ipt_end_row - ipt_start_row,
                ipt_end_col - ipt_start_col)
