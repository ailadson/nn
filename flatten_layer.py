class FlattenLayer():
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.next_layer = None
        prev_layer.next_layer = self
        self.rank1_shape = reduce(lambda acc, ele: acc * ele, self.prev_layer.output.shape)
        self.output = np.zeros(rank1_shape)

    def forward_propagate():
        self.output[:] = self.prev_layer.output.reshape(rank1_shape)[:]

    def back_propagate(self):
        pass

    def deriv_wrt_unit_outputs(self):
        next_deriv_wrt_unit_total_inputs = self.next_layer.deriv_wrt_unit_total_inputs()
        return next_deriv_wrt_unit_total_inputs.reshape(self.prev_layer.output.shape)
