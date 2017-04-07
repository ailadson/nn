# TODO: rewrite this Cython code in an AVX style!

cimport cython
cimport numpy as np
from tensor_utils cimport *

cdef DTYPE_t deconvolve2d_step(
        Rank2Tensor prev_output_layer,
        Rank2Tensor deriv_wrt_total_inputs_layer,
        int kernel_offset_i,
        int kernel_offset_j) nogil:

    cdef int img_rows = prev_output_layer.shape.dim0
    cdef int img_cols = prev_output_layer.shape.dim1
    cdef int prev_output_i, prev_output_j
    cdef int total_input_i, total_input_j
    cdef DTYPE_t val = 0.0

    for prev_output_i in range(img_rows):
        total_input_i = prev_output_i + kernel_offset_i
        if total_input_i < 0 or total_input_i >= img_rows:
            continue
        for prev_output_j in range(img_cols):
            total_input_j = prev_output_j + kernel_offset_j
            if total_input_j < 0 or total_input_j >= img_cols:
                continue
            val += (
                rank2_get(prev_output_layer, prev_output_i, prev_output_j) *
                rank2_get(deriv_wrt_total_inputs_layer, total_input_i, total_input_j)
            )

    return val

cdef void deconvolve2d_(
        Rank2Tensor prev_output_layer,
        Rank2Tensor deriv_wrt_weights_layer,
        Rank2Tensor deriv_wrt_total_inputs_layer) nogil:

    cdef int krows = deriv_wrt_weights_layer.shape.dim0
    cdef int kcols = deriv_wrt_weights_layer.shape.dim1
    cdef int kernel_i, kernel_j
    cdef int kernel_offset_i, kernel_offset_j
    cdef DTYPE_t val

    for kernel_i in range(krows):
        for kernel_j in range(kcols):
            kernel_offset_i = kernel_i - (krows // 2)
            kernel_offset_j = kernel_j - (kcols // 2)

            # += meant for accumulating errors in the deriv filter
            val = deconvolve2d_step(prev_output_layer,
                                    deriv_wrt_total_inputs_layer,
                                    kernel_offset_i,
                                    kernel_offset_j)
            rank2_inc(deriv_wrt_weights_layer, kernel_i, kernel_j, val)

cdef void deriv_wrt_weights_(
        Rank3Tensor prev_output_tensor,
        Rank4Tensor deriv_wrt_weights_tensor,
        Rank3Tensor deriv_wrt_total_inputs_tensor) nogil:

    cdef int prev_output_layer_idx
    cdef int total_input_layer_idx
    cdef int num_prev_output_layers = prev_output_tensor.shape.dim0
    cdef int num_total_input_layers = (
        deriv_wrt_total_inputs_tensor.shape.dim0
    )

    cdef Rank2Tensor prev_output_layer
    cdef Rank2Tensor deriv_wrt_weights_layer
    cdef Rank2Tensor deriv_wrt_total_inputs_layer

    for total_input_layer_idx in range(num_total_input_layers):
        for prev_output_layer_idx in range(num_prev_output_layers):
            prev_output_layer = rank3_offset(prev_output_tensor, prev_output_layer_idx)
            deriv_wrt_weights_layer = rank4_offset(
              deriv_wrt_weights_tensor,
              total_input_layer_idx,
              prev_output_layer_idx
            )
            deriv_wrt_total_inputs_layer = rank3_offset(
                deriv_wrt_total_inputs_tensor, total_input_layer_idx
            )

            deconvolve2d_(
                prev_output_layer,
                deriv_wrt_weights_layer,
                deriv_wrt_total_inputs_layer
            )

def deriv_wrt_weights(
        DTYPE_t[:, :, :] prev_output_layers,
        DTYPE_t[:, :, :, :] deriv_wrt_weights,
        DTYPE_t[:, :, :] deriv_wrt_unit_total_inputs):

    cdef Rank3Tensor prev_output_tensor = (
        memview_to_rank3_tensor(prev_output_layers)
    )
    cdef Rank4Tensor deriv_wrt_weights_tensor = (
        memview_to_rank4_tensor(deriv_wrt_weights)
    )
    cdef Rank3Tensor deriv_wrt_total_inputs_tensor = (
        memview_to_rank3_tensor(deriv_wrt_unit_total_inputs)
    )

    deriv_wrt_weights_(
        prev_output_tensor,
        deriv_wrt_weights_tensor,
        deriv_wrt_total_inputs_tensor
    )
