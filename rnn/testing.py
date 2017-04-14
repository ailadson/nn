import config
import graph
import tensorflow as tf
from text_data_set import TextDataSet


#build_graph
result = config.SAMPLE_PRIME
dataset = TextDataSet(f'../datasets/{config.DATASET_FILENAME}')
g = graph.build_graph(
    1, dataset.num_chars, config.NUM_LAYERS, 1, config.NUM_LSTM_UNITS
)

#restore session
def load_model(session):
    print("Start Load")
    if config.TEST_MODEL_FILENAME == config.LATEST_TEST_MODEL_FILENAME:
        model_path = tf.train.latest_checkpoint('./checkpoints')
    else:
        model_path = "./checkpoints/{config.LOAD_MODEL}"
    print(model_path)

    saver = tf.train.import_meta_graph(f"{model_path}.meta")
    saver.restore(session, f"{model_path}")
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in all_vars:
        v_ = session.run(v)
    print("Load Finished!")




with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    load_model(session)
    states = graph.make_initial_sampling_states(
        config.NUM_LAYERS, config.NUM_LSTM_UNITS
    )
    print(session.run([g.weights]))
    # feed in the initial sample string
    print("Start feeding in prime text")
    print(result)
    for char in result:
        char_encoding = dataset.char_to_one_hot(char)
        char_encoding = char_encoding.reshape([1, 1, -1])
        predictions, states = session.run(
            [g.all_predictions, g.final_states],
            feed_dict={
                g.inputs: char_encoding,
                tuple(g.initial_states): tuple(states)
            }
        )
    print("Finish Loading Prime")

    while len(result) < config.SAMPLE_LENGTH:
        prediction = predictions[0]
        char = dataset.one_hot_to_char(prediction)
        result += char
        states, predictions = session.run(
            [g.final_states, g.all_predictions],
            feed_dict={
                g.inputs: prediction.reshape([1,1,-1]),
                tuple(g.initial_states): tuple(states)
            }
        )

print(result)
