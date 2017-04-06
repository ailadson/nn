cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
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
@cython.wraparound(False)
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

cdef void backward_convolve2d_(
    np.float64_t[:, :] ipt,
    np.float64_t[:, :] kernel,
    np.float64_t[:, :] target) nogil:

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
    np.float64_t[:, :] ipt,
    np.float64_t[:, :] kernel,
    np.float64_t[:, :] target):

    convolve2d_(ipt, kernel, target)

def apply_convolution(
    np.float64_t[:, :, :] input_layers,
    np.float64_t[:, :, :, :] kernel_layers,
    np.float64_t[:, :, :] output_layers):

    cdef int input_layer_idx
    cdef int num_input_layers = input_layers.shape[0]
    cdef int output_layer_idx
    cdef int num_output_layers = output_layers.shape[0]

    for output_layer_idx in range(num_output_layers):
        for input_layer_idx in range(num_input_layers):
            input_layer = input_layers[input_layer_idx]
            kernel_layer = kernel_layers[output_layer_idx, input_layer_idx]
            output_layer = output_layers[output_layer_idx]
            convolve2d_(input_layer, kernel_layer, output_layer)

def apply_backwards_convolution(
        np.float64_t[:, :, :] deriv_wrt_total_inputs,
        np.float64_t[:, :, :, :] kernel_layers,
        np.float64_t[:, :, :] deriv_wrt_prev_outputs):

    cdef int input_layer_idx
    cdef int num_input_layers = deriv_wrt_prev_outputs.shape[0]
    cdef int output_layer_idx
    cdef int num_output_layers = deriv_wrt_total_inputs.shape[0]

    for output_layer_idx in range(num_output_layers):
        for input_layer_idx in range(num_input_layers):
            total_input_layer = deriv_wrt_total_inputs[output_layer_idx]
            kernel_layer = kernel_layers[output_layer_idx, input_layer_idx]
            prev_output_layer = deriv_wrt_prev_outputs[input_layer_idx]

            backward_convolve2d_(
                total_input_layer,
                kernel_layer,
                prev_output_layer
            )
