cimport cython
cimport numpy as np

ctypedef np.float64_t DTYPE_t

cdef struct s_int_pair:
    int i
    int j

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
    for i in range(width):
        for j in range(height):
            get_local_max_and_pos(
                channel_idx, prev_channels, i, j, &max_pair
            )
            deriv_wrt_prev_outputs[channel_idx, max_pair.i, max_pair.j] = (
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
    s_int_pair* max_pair,
    DTYPE_t* max_val) nogil:

    if good and channels[channel_idx, i, j] > max_val[0]:
        max_pair.i = i
        max_pair.j = j
        max_val[0] = channels[channel_idx, i, j]

# One fn.
# Looks at each sell as in getpoolingvalues. Keeps track of max seen so far and its pos.
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
    cdef DTYPE_t max_val = channels[channel_idx, i, j]

    (max_pair).i = i
    (max_pair).j = j

    perform(good_j, channels, channel_idx, i, j + 1, max_pair, &max_val)
    perform(good_i, channels, channel_idx, i + 1, j, max_pair, &max_val)
    perform(good_i & good_j, channels, channel_idx, i + 1, j + 1, max_pair, &max_val)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_max_pooling_(
    DTYPE_t[:, :, :] ipt,
    DTYPE_t[:, :, :] des) nogil:
    cdef int num_channels = ipt.shape[0]
    cdef int height = ipt.shape[1]
    cdef int width = ipt.shape[2]
    cdef int channel_idx, i, j
    cdef np.float64_t curr_val, next_val

    for channel_idx in range(num_channels):
        # Compare to item in next column.
        for i in range(height):
            for j in range(width - 1):
                curr_val = ipt[channel_idx, i, j]

                next_val = ipt[channel_idx, i, j + 1]
                if next_val > curr_val:
                    des[channel_idx, i, j] = next_val
                else:
                    des[channel_idx, i, j] = curr_val

        # Compare to item in next row.
        for i in range(height - 1):
            for j in range(width):
                # Notice that I use des because this is already the
                # max of looking one right.
                curr_val = des[channel_idx, i, j]
                next_val = des[channel_idx, i + 1, j]
                if next_val > curr_val:
                    des[channel_idx, i, j] = next_val
                else:
                    des[channel_idx, i, j] = curr_val

def apply_max_pooling(
    DTYPE_t[:, :, :] ipt,
    DTYPE_t[:, :, :] des):

    apply_max_pooling_(ipt, des)
