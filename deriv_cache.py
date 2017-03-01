from functions import zeros

class DerivativeCache:
    def __init__(self, layer):
        self.weights = zeros([layer.num_units, layer.prev_layer.num_units])
        self.unit_outputs = zeros([layer.num_units])
        self.unit_total_inputs = zeros([layer.num_units])
        self.reset()

    def reset(self):
        self.is_set = { "weights" : False,
                        "unit_outputs" : False,
                        "unit_total_inputs" : False
                    }

    def is_set(self, name):
        return self.is_set[name]
