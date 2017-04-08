class Layer:
    def __init__(self, prev_layer, num_units, activation_func_name):
        self.num_units = num_units

        activation_funcs = get_activation_functions(activation_func_name)
        self.activation_func = activation_funcs[0]
        self.deriv_activation_func = activation_funcs[1]

        self.prev_layer = prev_layer
        self.next_layer = None
        if prev_layer is not None:
            prev_layer.next_layer = self

        self.total_input = np.zeros([num_units], dtype=np.float32)
        self.output = np.zeros([num_units], dtype=np.float32)

    def back_propagate(self):
        raise Exception("Not Implemented!")

    def forward_propagate(self):
        raise Exception("Not Implemented!")

    def has_weights(self):
        raise Exception("Not Implemented!")

def get_activation_functions(name):
    if name == 'id':
        activation_func = lambda val, des: np.copyto(des, val)
        deriv_activation_func = lambda val, des: des.fill(1)
    elif name == 'relu':
        activation_func = relu
        deriv_activation_func = derivative_of_relu
    elif name == 'sigmoid':
        activation_func = sigmoid
        deriv_activation_func = derivative_of_sigmoid
    else:
        raise Exception("No Activation Function Given!")

    return (activation_func, deriv_activation_func)
