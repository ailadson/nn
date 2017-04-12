from config import *
import tensorflow as tf
from graph import build_graph
from file_reader import FileReader
from batcher import *

fr = FileReader('../datasets/test.txt')
batches = make_batches(fr.one_hot, BATCH_SIZE, STEP_SIZE)
train_batches, val_batches = partition_batches(batches)

g = build_graph(BATCH_SIZE, fr.num_chars, NUM_LAYERS, STEP_SIZE, NUM_LSTM_UNITS)
saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(EPOCHS):
    states = [tf.contrib.rnn.LSTMStateTuple(
        np.zeros([BATCH_SIZE, NUM_LSTM_UNITS]),
        np.zeros([BATCH_SIZE, NUM_LSTM_UNITS])
    ) for _ in range(NUM_LAYERS)]

    for j, (batch_x, batch_y) in enumerate(train_batches):
        states, loss, _ = session.run([g.final_states, g.avg_loss, g.train_op], feed_dict={
            g.inputs: batch_x,
            g.outputs: batch_y,
            tuple(g.initial_states): tuple(states)
        })
        if j % 10 == 0:
            print(f"Epoch {i}, Batch {j} | Loss: {loss}")

    avg_loss = 0
    print("Starting Validation")
    for j, (batch_x, batch_y) in enumerate(val_batches):
        states, loss = session.run([g.final_states, g.avg_loss], feed_dict={
            g.inputs: batch_x,
            g.outputs: batch_y,
            tuple(g.initial_states): tuple(states)
        })
        avg_loss += loss
    avg_loss /= len(val_batches)
    saver.save(session, f"checkpoints/my-{i}_test_{avg_loss}")
    print(f"Epoch {i} | Validation Loss: {avg_loss}")
