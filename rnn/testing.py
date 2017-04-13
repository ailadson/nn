import config
import graph

#build_graph
result = config.SAMPLE_PRIME
saver = tf.train.Saver()
dataset = TextDataSet(f'../datasets/{config.DATASET_FILENAME}')
g = graph.build_graph(
    1, dataset.num_chars_in_text, config.NUM_LAYERS, 1, config.NUM_LSTM_UNITS
)

#restore session
def load_model(session):
    if config.LOAD_MODEL == config.LOAD_LATEST:
        saver.restore(
            session,
            f"./checkpoints/{tf.train.latest_checkpoint('./checkpoints')}"
        )
    else:
        saver.restore(
            session,
            f"./checkpoints/{config.LOAD_MODEL}"
        )




with tf.Session() as session:
    load_model(session)
    states = graph.make_initial_sampling_states(
        config.NUM_LAYERS, config.NUM_LSTM_UNITS
    )

    # feed in the initial sample string
    for char in result:
        char_encoding = dataset.char_to_one_hot(char)
        char_encoding = char_encoding.reshape([-1, -1, dataset.num_chars])
        states, predictions = session.run(
            [g.final_states, g.all_predictions],
            feed_dict={
                g.inputs: char_encoding,
                tuple(g.initial_states): tuple(states)
            }
        )

    while len(result) < config.SAMPLE_LENGTH:
        prediction = predictions[0]
        char = dataset.one_hot_to_char(prediction)
        result += char
        states, predictions = session.run(
            [g.final_states, g.all_predictions],
            feed_dict={
                g.inputs: char_encoding,
                tuple(g.initial_states): tuple(states)
            }
        )


#run code to sample the data
