cimport cython
cimport numpy as np
from numpy.math cimport INFINITY

ctypedef np.float32_t DTYPE_t

cdef struct s_int_pair:
    int i
    int j
    DTYPE_t val

def back_propagate_channels(
        deriv_wrt_prev_outputs,
        prev_channels,
        deriv_wrt_unit_outputs):

    back_propagate_channels_(
        deriv_wrt_prev_outputs,
        prev_channels,
        deriv_wrt_unit_outputs
    )

cdef void back_propagate_channels_(
    DTYPE_t[:, :, :] deriv_wrt_prev_outputs,
    DTYPE_t[:, :, :] prev_channels,
    DTYPE_t[:, :, :] deriv_wrt_unit_outputs):

    cdef int num_of_channels = prev_channels.shape[0]
    cdef int i
    for i in range(num_of_channels):
        back_propagate_channel(
            i,
            deriv_wrt_prev_outputs,
            prev_channels,
            deriv_wrt_unit_outputs
        )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void back_propagate_channel(
    int channel_idx,
    DTYPE_t[:, :, :] deriv_wrt_prev_outputs,
    DTYPE_t[:, :, :] prev_channels,
    DTYPE_t[:, :, :] derive_wrt_channel_output) nogil:

    cdef int i, j
    cdef int width = derive_wrt_channel_output.shape[1]
    cdef int height = derive_wrt_channel_output.shape[2]
    cdef s_int_pair max_pair
    cdef int max_i, max_j

    for i in range(width):
        for j in range(height):
            get_local_max_and_pos(
                channel_idx, prev_channels, i, j, &max_pair
            )
            max_i = max_pair.i
            max_j = max_pair.j
            deriv_wrt_prev_outputs[channel_idx, max_i, max_j] = (
                derive_wrt_channel_output[channel_idx, i, j]
            )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform(
    int good,
    DTYPE_t[:, :, :] channels,
    int channel_idx,
    int i,
    int j,
    s_int_pair* max_pair) nogil:

    cdef DTYPE_t val = channels[channel_idx, i, j]
    if good and val > max_pair.val:
        max_pair.i = i
        max_pair.j = j
        max_pair.val = val

# One fn. Looks at each sell as in getpoolingvalues. Keeps track of
# max seen so far and its pos.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_local_max_and_pos(
    int channel_idx,
    DTYPE_t[:, :, :] channels,
    int block_i,
    int block_j,
    s_int_pair* max_pair) nogil:

    cdef int i = block_i * 2
    cdef int j = block_j * 2
    cdef int good_i = (i + 1 < channels.shape[1])
    cdef int good_j = (j + 1 < channels.shape[2])

    max_pair.i = i
    max_pair.j = j
    max_pair.val = channels[channel_idx, i, j]

    perform(good_j, channels, channel_idx, i, j + 1, max_pair)
    perform(good_i, channels, channel_idx, i + 1, j, max_pair)
    perform(
        good_i and good_j,
        channels,
        channel_idx,
        i + 1,
        j + 1,
        max_pair
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_max_pooling_(
    DTYPE_t[:, :, :] ipt,
    DTYPE_t[:, :, :] des) nogil:
    cdef int num_channels = ipt.shape[0]
    cdef int height = ipt.shape[1]
    cdef int width = ipt.shape[2]
    cdef int channel_idx, i, j
    cdef DTYPE_t curr_val, next_val

    des[:, :, :] = -INFINITY

    for channel_idx in range(num_channels):
        for i in range(height):
            for j in range(width):
                curr_val = ipt[channel_idx, i, j]
                if curr_val > des[channel_idx, i // 2, j // 2]:
                    des[channel_idx, i // 2, j // 2] = curr_val

def apply_max_pooling(
    DTYPE_t[:, :, :] ipt,
    DTYPE_t[:, :, :] des):

    apply_max_pooling_(ipt, des)
