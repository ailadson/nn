import tensorflow as tf
from collections import namedtuple

LEARNING_RATE = 0.001
CLIP_GRADIENT = 5

Graph = namedtuple('Graph', [
    'inputs', 'outputs', 'initial_state',
    'final_state', 'all_predictions', 'total_loss', 'train_op'
])

def build_graph(batch_size, num_chars, num_layers, num_time_steps, num_lstm_units):
    inputs = tf.placeholder(tf.float32, [batch_size, num_time_steps, num_chars])
    outputs = tf.placeholder(tf.float32, [batch_size, num_time_steps, num_chars])

    cell = tf.contrib.rnn.BasicLSTMCell(num_lstm_units, state_is_tuple=True)
    # state = initial_state = tf.placeholder(tf.float32, [batch_size, num_lstm_units])
    state = initial_state = cell.zero_state(batch_size, tf.float32)

    weight_mat = tf.Variable(tf.truncated_normal([num_lstm_units, num_chars]))
    bias_vec = tf.Variable(tf.zeros([num_chars]))
    all_logits = []
    all_predictions = []

    for i in range(num_time_steps):
        output, state = cell(inputs[:, i, :], state)
        logits = tf.matmul(output, weight_mat) + bias_vec
        all_logits.append(logits)
        all_predictions.append(tf.nn.softmax(logits))

    final_state = state

    total_loss = 0
    for i, logits in enumerate(all_logits):
        total_loss += tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=outputs[:, i, :]
        )
    total_loss /= batch_size

    tvars = tf.trainable_variables()
    grads = tf.gradients(total_loss, tvars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, CLIP_GRADIENT)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.apply_gradients(clipped_grads)

    return Graph(
        inputs=inputs,
        outputs=outputs,
        initial_state=initial_state,
        final_state=final_state,
        all_predictions=all_predictions,
        total_loss=total_loss,
        train_op=train_op
    )
