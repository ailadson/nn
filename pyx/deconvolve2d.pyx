# TODO: rewrite this Cython code in an AVX style!

cimport cython
cimport numpy as np
from tensor_utils cimport *

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
    Rank2Shape shape, int offset_row, int offset_col) nogil:

    cdef int num_of_rows = shape.dim0
    cdef int num_of_cols = shape.dim1
    cdef bounds_s bounds

    bounds.start_row = max(offset_row, 0)
    bounds.end_row = min(offset_row + num_of_rows, num_of_rows)
    bounds.start_col = max(offset_col, 0)
    bounds.end_col = min(offset_col + num_of_cols, num_of_cols)

    return bounds

cdef DTYPE_t sum_el_prod2d(
      Rank2Tensor ipt,
      Rank2Tensor err,
      bounds_s ipt_bounds,
      bounds_s err_bounds) nogil:

    cdef DTYPE_t s = 0.0
    cdef DTYPE_t ipt_el, err_el
    cdef int num_rows = ipt_bounds.end_row - ipt_bounds.start_row
    cdef int num_cols = ipt_bounds.end_col - ipt_bounds.end_col
    cdef int i, j

    for i in range(num_rows):
        for j in range(num_cols):
            ipt_el = rank2_get(
                ipt, ipt_bounds.start_row + i, ipt_bounds.start_col + j
            )
            err_el = rank2_get(
                err, err_bounds.start_row + i, err_bounds.start_col + j
            )

            s += ipt_el * err_el

    return s

cdef void deconvolve2d_(
        Rank2Tensor ipt,
        Rank2Tensor error,
        Rank2Tensor deriv_filter) nogil:

    cdef int krows = deriv_filter.shape.dim0
    cdef int kcols = deriv_filter.shape.dim1
    cdef int i, j
    cdef int offset_row, offset_col
    cdef bounds_s ipt_bounds
    cdef bounds_s error_bounds
    cdef DTYPE_t val

    for i in range(krows):
        for j in range(kcols):
            offset_row = i - (krows // 2)
            offset_col = j - (kcols // 2)
            ipt_bounds = get_deconvolve_bounds(
                ipt.shape, offset_row, offset_col
            )
            error_bounds = get_deconvolve_bounds(
                error.shape, -1 * offset_row, -1 * offset_col
            )

            # += meant for accumulating errors in the deriv filter
            val = sum_el_prod2d(
                ipt, error, ipt_bounds, error_bounds
            )
            rank2_inc(deriv_filter, i, j, val)

cdef void deriv_wrt_weights_(
        Rank3Tensor input_tensor,
        Rank4Tensor deriv_wrt_weights_tensor,
        Rank3Tensor deriv_wrt_total_inputs_tensor) nogil:

    cdef int input_layer_idx
    cdef int num_output_layers = (
        deriv_wrt_total_inputs_tensor.shape.dim0
    )
    cdef int output_layer_idx
    cdef int num_input_layers = input_tensor.shape.dim0

    cdef Rank2Tensor input_layer
    cdef Rank2Tensor deriv_wrt_total_inputs_layer
    cdef Rank2Tensor deriv_wrt_weights_layer

    for output_layer_idx in range(num_output_layers):
        for input_layer_idx in range(num_input_layers):
            input_layer = rank3_offset(input_tensor, input_layer_idx)
            deriv_wrt_total_inputs_layer = rank3_offset(
                deriv_wrt_total_inputs_tensor, output_layer_idx
            )
            deriv_wrt_weights_layer = rank4_offset(
                deriv_wrt_weights_tensor,
                output_layer_idx,
                input_layer_idx
            )

            deconvolve2d_(
                input_layer,
                deriv_wrt_total_inputs_layer,
                deriv_wrt_weights_layer
            )

def deriv_wrt_weights(
        DTYPE_t[:, :, :] input_layers,
        DTYPE_t[:, :, :, :] deriv_wrt_weights,
        DTYPE_t[:, :, :] deriv_wrt_unit_total_inputs):

    cdef Rank3Tensor input_tensor = (
        memview_to_rank3_tensor(input_layers)
    )
    cdef Rank4Tensor deriv_wrt_weights_tensor = (
        memview_to_rank4_tensor(deriv_wrt_weights)
    )
    cdef Rank3Tensor deriv_wrt_total_inputs_tensor = (
        memview_to_rank3_tensor(deriv_wrt_unit_total_inputs)
    )

    deriv_wrt_weights_(
        input_tensor,
        deriv_wrt_weights_tensor,
        deriv_wrt_total_inputs_tensor
    )
