# from net import *
# from functions import *
import numpy as np

observed = np.ones([4, 4]) * 2
data = np.ones([4, 4]) * 4
estimate = np.array([16,24,24,16,24,36,36,24,24,36,36,24,16,24,24,16]).reshape([4, 4])
print(4 * ((observed - estimate) * -2).sum())
#
#
# def assert_array_eql(arr1, arr2):
#     for i in range(len(arr1)):
#         assert(abs(arr1[i] - arr2[i]) < 0.0001)
#
# def forward_propagate():
#     nn = Net(2)
#     nn.add_layer(2)
#     nn.layers[-1].weights = np.array([[1.0,-1.0],[-3.0,4.0]])
#     nn.add_output_layer(1)
#     nn.layers[-1].weights = np.array([[1.0, -1.0]])
#     output = nn.forward_propagate(np.array([1.0,2.0]))
#
#     assert_array_eql(output, [0.3264323453466589])
#     assert_array_eql(nn.layers[1].output, [0.2689414213699951, 0.9933071490757153])
#
# def back_propagate():
#     nn = Net(2)
#     nn.add_layer(2)
#     nn.layers[-1].weights = np.array([[1.0,-1.0],[-3.0,4.0]])
#     nn.add_output_layer(1)
#     nn.layers[-1].weights = np.array([[1.0, -1.0]])
#
#     nn.forward_propagate(np.array([1.0,2.0]))
#     nn.back_propagate(1.0)
#
#     assert_array_eql(nn.layers[-1].deriv_cache.unit_outputs, [-3.0634219134688925])
#
#     expected_total_input = 0.2689414213699951 - 0.9933071490757153
#     expected_derivative_of_total_input = derivative_of_sig(expected_total_input)
#     expected_derivative_of_total_input *= -3.0634219134688925
#     assert_array_eql([expected_derivative_of_total_input], nn.layers[-1].deriv_cache.unit_total_inputs)
#
#     expected_deriv_wrt_weights = [[0.2689414213699951 * expected_derivative_of_total_input, 0.9933071490757153 * expected_derivative_of_total_input]]
#     assert_array_eql(expected_deriv_wrt_weights[0], nn.layers[-1].deriv_cache.weights[0])
#
#
# forward_propagate()
# back_propagate()


# def pad_flipped_weights(weights):
#     height, width = weights.shape
#     new_height = height + (1 if height % 2 == 0 else 0)
#     new_width = width + (1 if width % 2 == 0 else 0)
#     if height % 2 == 0:
#
#         weights = np.append(weights, np.zeros([1, width]), axis=0)
#     if width % 2 == 0:
#         weights = np.append(weights, np.zeros([new_height, 1]), axis=1)
#     return weights
#
# b = np.arange(1, 10).reshape([3, 3])
# print(pad_flipped_weights(b))
