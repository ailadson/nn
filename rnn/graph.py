import tensorflow as tf
from collections import namedtuple

LEARNING_RATE = 0.001
CLIP_GRADIENT = 5

Graph = namedtuple('Graph', [
    'inputs', 'outputs', 'initial_states',
    'final_states', 'all_predictions', 'avg_loss', 'train_op'
])

def build_graph(
    batch_size,
    num_chars,
    num_layers,
    num_time_steps,
    num_lstm_units,
    predict=False):

    if predict:
        batch_size = 1
        num_time_steps = 1

    inputs = tf.placeholder(tf.float32, [batch_size, num_time_steps, num_chars])
    outputs = tf.placeholder(tf.float32, [batch_size, num_time_steps, num_chars])

    cells = []
    states = []

    for i in range(num_layers):
        cells.append(
            tf.contrib.rnn.BasicLSTMCell(num_lstm_units, state_is_tuple=True)
        )
        states.append(
            tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder(tf.float32, [batch_size, num_lstm_units]),
                tf.placeholder(tf.float32, [batch_size, num_lstm_units])
            )
        )

    initial_states = states[:]

    weight_mat = tf.Variable(tf.truncated_normal([num_lstm_units, num_chars]))
    bias_vec = tf.Variable(tf.zeros([num_chars]))
    all_logits = []
    all_predictions = []

    for i in range(num_time_steps):
        next_input = inputs[:, i, :]

        for j in range(num_layers):
            with tf.variable_scope(f'layers{j}') as scope:
                if i > 0:
                    scope.reuse_variables()
                next_input, next_state = cells[j](next_input, states[j])
                states[j] = next_state
        logits = tf.matmul(next_input, weight_mat) + bias_vec
        all_logits.append(logits)
        all_predictions.append(tf.nn.softmax(logits))

    final_states = states

    avg_loss = 0
    for i, logits in enumerate(all_logits):
        avg_loss += tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=outputs[:, i, :]
            )
        )
    avg_loss /= num_time_steps

    tvars = tf.trainable_variables()
    grads = tf.gradients(avg_loss, tvars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, CLIP_GRADIENT)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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
