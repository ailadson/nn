import tensorflow as tf
from graph import build_graph
from file_reader import FileReader
from batcher import *

BATCH_SIZE = 10
STEP_SIZE = 50
NUM_LAYERS = 1
NUM_LSTM_UNITS = 128
EPOCHS = 1

fr = FileReader('../datasets/anna.txt')
batches = make_batches(fr.one_hot, BATCH_SIZE, STEP_SIZE)
train_batches, val_batches = partition_batches(batches)

g = build_graph(BATCH_SIZE, fr.num_chars, NUM_LAYERS, STEP_SIZE, NUM_LSTM_UNITS)


session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(EPOCHS):
    state = tf.contrib.rnn.LSTMStateTuple(
        np.zeros([BATCH_SIZE, NUM_LSTM_UNITS]),
        np.zeros([BATCH_SIZE, NUM_LSTM_UNITS])
    )

    for j, (batch_x, batch_y) in enumerate(train_batches):
        state, loss, _ = session.run([g.final_state, g.avg_loss, g.train_op], feed_dict={
            g.inputs: batch_x,
            g.outputs: batch_y,
            g.initial_state.c: state.c,
            g.initial_state.h: state.h
        })
        print(f"Epoch {i}, Batch {j} | Loss: {loss}")
