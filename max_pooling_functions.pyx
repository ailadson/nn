import math
cimport numpy as np

def back_propagate_channels(
    np.ndarray[np.float64_t, ndim=3] deriv_wrt_prev_outputs,
    np.ndarray[np.float64_t, ndim=3] prev_channels,
    np.ndarray[np.float64_t, ndim=3] deriv_wrt_unit_outputs):

    cdef int num_of_channels = prev_channels.shape[0]
    cdef int i
    for i in range(num_of_channels):
        back_propagate_channel(i, deriv_wrt_prev_outputs, prev_channels[i], deriv_wrt_unit_outputs[i])

cdef back_propagate_channel(
    int channel_idx,
    np.ndarray[np.float64_t, ndim=3] deriv_wrt_prev_outputs,
    np.ndarray[np.float64_t, ndim=2] prev_channel,
    np.ndarray[np.float64_t, ndim=2] derive_wrt_channel_output):

    cdef int i, j
    cdef int width = derive_wrt_channel_output.shape[0]
    cdef int height = derive_wrt_channel_output.shape[1]
    for i in range(width):
        for j in range(height):
            _, max_i, max_j = get_local_max_and_pos(prev_channel, i, j)
            deriv_wrt_prev_outputs[channel_idx, max_i, max_j] = derive_wrt_channel_output[i, j]

# One fn.
# Looks at each sell as in getpoolingvalues. Keeps track of max seen so far and its pos.
cdef get_local_max_and_pos(np.ndarray[np.float64_t, ndim=2] channel, int block_i, int block_j):
    cdef int i = block_i * 2
    cdef int j = block_j * 2
    cdef int good_i = (i + 1 < channel.shape[0])
    cdef int good_j = (j + 1 < channel.shape[1])
    cdef int max_i = i
    cdef int max_j = j
    cdef np.float64_t max_val = channel[i, j]

    if good_j and channel[i, j + 1] > max_val:
        max_val = channel[i, j + 1]
        max_i = i
        max_j = j
    if good_i and channel[i + 1, j] > max_val:
        max_val = channel[i + 1, j]
        max_i = i + 1
        max_j = j
    if good_i and good_j and channel[i + 1, j + 1] > max_val:
        max_val = channel[i + 1, j + 1]
        max_i = i + 1
        max_j = j + 1

    return (max_val, max_i, max_j)
