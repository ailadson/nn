from collections import namedtuple
import config
import tensorflow as tf
import numpy as np


Graph = namedtuple('Graph', [
    'inputs', 'outputs', 'initial_states',
    'final_states', 'all_predictions', 'avg_loss', 'train_op'
])

def build_graph(
    batch_size,
    num_chars,
    num_layers,
    num_time_steps,
    num_lstm_units):

    inputs = tf.placeholder(
        tf.float32, [batch_size, num_time_steps, num_chars],
        name="inputs"
    )
    outputs = tf.placeholder(
        tf.float32, [batch_size, num_time_steps, num_chars],
        name="outputs"
    )

    cells = [
        tf.contrib.rnn.BasicLSTMCell(
            num_lstm_units, state_is_tuple=True
        ) for i in range(num_layers)
    ]

    prev_states = [
        tf.contrib.rnn.LSTMStateTuple(
            tf.placeholder(
                tf.float32,
                [batch_size, num_lstm_units],
                name=f"initial_state/{i}/c"
            ), tf.placeholder(
                tf.float32,
                [batch_size, num_lstm_units],
                name=f"initial_state/{i}/h"
            )
        ) for i in range(num_layers)
    ]

    initial_states = prev_states[:]

    weight_mat = tf.Variable(
        tf.truncated_normal([num_lstm_units, num_chars]),
        name="softmax_weights"
    )
    bias_vec = tf.Variable(
        tf.zeros([num_chars]),
        name="softmax_weights"
    )
    all_logits = []
    all_predictions = []

    for time_step in range(num_time_steps):
        prev_output = inputs[:, time_step, :]

        for layer_idx in range(num_layers):
            with tf.variable_scope(f'layers/{layer_idx}') as scope:
                if time_step > 0:
                    scope.reuse_variables()

                prev_output, prev_states[layer_idx] = (
                    cells[layer_idx](
                        prev_output, prev_states[layer_idx]
                    )
                )

        logits = tf.matmul(prev_output, weight_mat) + bias_vec
        all_logits.append(logits)
        all_predictions.append(tf.nn.softmax(logits))

    final_states = prev_states

    avg_loss = 0
    classification_rate = 0
    for i, logits in enumerate(all_logits):
        avg_loss += tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=outputs[:, i, :]
            )
        )
        # correct_predictions = tf.equal(
        #     tf.argmax(logits, 1), tf.argmax(outputs[:, i, :])
        # )
        # classification_rate = tf.reduce_mean(
        #     tf.cast(correct_predictions, "float")
        # )

    avg_loss /= num_time_steps
    # classification_rate /= num_time_steps

    tvars = tf.trainable_variables()
    grads = tf.gradients(avg_loss, tvars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, config.CLIP_GRADIENT)
    optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE)
    train_op = optimizer.apply_gradients(zip(clipped_grads, tvars))

    return Graph(
        inputs=inputs,
        outputs=outputs,
        initial_states=initial_states,
        final_states=final_states,
        all_predictions=all_predictions,
        avg_loss=avg_loss,
        train_op=train_op
    )

def make_initial_states(batch_size, num_layers, num_lstm_units):
    return [
        tf.contrib.rnn.LSTMStateTuple(
            np.zeros([batch_size, num_lstm_units], dtype=np.float32),
            np.zeros([batch_size, num_lstm_units], dtype=np.float32)
        ) for _ in range(num_layers)
    ]

def build_sampling_graph(
        num_chars,
        num_layers,
        num_lstm_units):
    return build_graph(
        1, num_chars, num_layers, 1, num_lstm_units
    )

def make_initial_sampling_states(num_layers, num_lstm_units):
    return make_initial_states(1, num_layers, num_lstm_units)
