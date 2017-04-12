import tensorflow as tf
import numpy as np

def predict(session, g, fr, prime, prediction_length, num_layers, num_lstm_units):
    states = [tf.contrib.rnn.LSTMStateTuple(
        np.zeros([1, num_lstm_units]),
        np.zeros([1, num_lstm_units])
    ) for _ in range(num_layers)]

    for char in prime:
        char_encoding = fr.char_to_one_hot(char).reshape([1, 1, fr.num_chars])
        states, prediction = session.run([g.final_states, g.all_predictions], feed_dict={
            g.inputs: char_encoding,
            tuple(g.initial_states): tuple(states)
        })

    print("Initial State Set!")

    while len(prime) < prediction_length:
        next_char = fr.one_hot_to_char(prediction[0].reshape([fr.num_chars]))
        prime += next_char
        char_encoding = fr.char_to_one_hot(next_char).reshape([1, 1, fr.num_chars])

        states, prediction = session.run([g.final_states, g.all_predictions], feed_dict={
            g.inputs: char_encoding,
            tuple(g.initial_states): tuple(states)
        })

    return prime
