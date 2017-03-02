from net import *
from functions import *

def assert_array_eql(arr1, arr2):
    for i in range(len(arr1)):
        assert(abs(arr1[i] - arr2[i]) < 0.0001)

def forward_propagate():
    nn = Net(2)
    nn.add_layer(2)
    nn.layers[-1].weights = [[1,-1],[-3,4]]
    nn.add_output_layer(1)
    nn.layers[-1].weights = [[1, -1]]
    output = nn.forward_propagate([1,2])

    assert_array_eql(output, [0.3264323453466589])
    assert_array_eql(nn.layers[1].output, [0.2689414213699951, 0.9933071490757153])

def back_propagate():
    nn = Net(2)
    nn.add_layer(2)
    nn.layers[-1].weights = [[1,-1],[-3,4]]
    nn.add_output_layer(1)
    nn.layers[-1].weights = [[1, -1]]

    nn.forward_propagate([1,2])
    nn.back_propagate(1)

    print(nn.layers[-1].deriv_cache)
    assert_array_eql(nn.layers[-1].deriv_cache.unit_outputs, [-3.0634219134688925])

    expected_total_input = 0.2689414213699951 - 0.9933071490757153
    expected_derivative_of_total_input = derivative_of_sig(expected_total_input)
    expected_derivative_of_total_input *= -3.0634219134688925
    assert_array_eql([expected_derivative_of_total_input], nn.layers[-1].deriv_cache.unit_total_inputs)

    expected_deriv_wrt_weights = [[0.2689414213699951 * expected_derivative_of_total_input, 0.9933071490757153 * expected_derivative_of_total_input]]
    assert_array_eql(expected_deriv_wrt_weights[0], nn.layers[-1].deriv_cache.weights[0])


forward_propagate()
back_propagate()
