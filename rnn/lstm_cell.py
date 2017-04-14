import tensorflow as tf

class BasicLSTMCell:
    def __init__(self, num_lstm_units, num_input_units, name):
        self.forget_weights = tf.Variable(
            tf.random_uniform([num_lstm_units + num_input_units, num_lstm_units],
            minval=-1,
            maxval=1),
            name=f"{name}/lstm_forget_weights"
        )
        self.forget_biases = tf.Variable(
            tf.ones([num_lstm_units]),
            name=f"{name}/lstm_forget_biases"
        )
        self.write_weights = tf.Variable(
            tf.random_uniform([num_lstm_units + num_input_units, num_lstm_units],
            minval=-1,
            maxval=1),
            name=f"{name}/lstm_write_weights"
        )
        self.write_biases = tf.Variable(
            tf.zeros([num_lstm_units]),
            name=f"{name}/lstm_write_biases"
        )
        self.write_modulation_weights = tf.Variable(
            tf.random_uniform([num_lstm_units + num_input_units, num_lstm_units],
            minval=-1,
            maxval=1),
            name=f"{name}/lstm_write_modulation_weights"
        )
        self.write_modulation_biases = tf.Variable(
            tf.zeros([num_lstm_units]),
            name=f"{name}/lstm_write_modulation_biases"
        )
        self.read_weights = tf.Variable(
            tf.random_uniform([num_lstm_units + num_input_units, num_lstm_units],
            minval=-1,
            maxval=1),
            name=f"{name}/lstm_read_weights"
        )
        self.read_biases = tf.Variable(
            tf.zeros([num_lstm_units]),
            name=f"{name}/lstm_read_biases"
        )

    def __call__(self, ipt, prev_output_and_prev_state):
        def forget_gate_multiplier():
            return get_value(
                self.forget_weights, self.forget_biases
            )

        def write_gate_multiplier():
            return get_value(self.write_weights, self.write_biases)

        def write_modulation_gate_value():
            z = tf.matmul(
                tf.concat([ipt, prev_output], 1), self.write_modulation_weights
            ) + self.write_modulation_biases
            return tf.tanh(z)

        def read_gate_multiplier():
            return get_value(self.read_weights, self.read_biases)

        def get_value(weights, biases):
            weights /= 255
            z = tf.matmul(tf.concat([ipt, prev_output], 1), weights) + biases
            return tf.sigmoid(z)

        prev_output, prev_state = prev_output_and_prev_state

        next_state = (
            prev_state * forget_gate_multiplier() +
            write_modulation_gate_value() * write_gate_multiplier()
        )

        next_output = (
            tf.tanh(next_state) * read_gate_multiplier()
        )

        return (next_output, (next_output, next_state))
