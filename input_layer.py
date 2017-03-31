class InputLayer:
    def __init__(self, num_units):
        self.output = None
        self.num_units = num_units

    def set_input(self, input_v):
        if len(input_v) != self.num_units:
            raise Exception(f"Wrong number of units in input: {len(input_v)} | Expected: {self.num_units}")
        self.output = input_v

    def forward_propagate(self):
        pass

    def has_weights(self):
        return False
