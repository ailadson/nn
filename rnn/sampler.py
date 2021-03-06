import config
import graph
import tensorflow as tf
from text_data_set import TextDataSet

BATCH_SIZE = 1
STEP_SIZE = 1

def load_model(session):
    print("Start Load")
    if config.TEST_MODEL_FILENAME == config.LATEST_TEST_MODEL_FILENAME:
        model_path = tf.train.latest_checkpoint('checkpoints')
    else:
        model_path = "checkpoints/{config.LOAD_MODEL}"
    # To confirm the correct file will be loaded
    print(model_path)

    # Saver() causes an error. Stack overflow gave me this solution
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(f"{model_path}.meta")
    saver.restore(session, f"{model_path}")
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in tf.global_variables():
        v_ = session.run(v)
        # if v.name is config.TEST_VAR_NAME: print(v_)
    print("Load Finished!")


def create_sample(session, g, dataset, states):
    result = config.SAMPLE_PRIME
    print(f"Start Feeding In Prime Text: {result}")
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

    print("Producing Sample...")
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
    return result

def run():
    dataset = TextDataSet(f'../datasets/{config.DATASET_FILENAME}')
    g = graph.build_graph(
        BATCH_SIZE, dataset.num_chars, config.NUM_LAYERS, STEP_SIZE, config.NUM_LSTM_UNITS
    )
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        load_model(session)
        states = graph.make_initial_sampling_states(
            config.NUM_LAYERS, config.NUM_LSTM_UNITS
        )
        return create_sample(session, g, dataset, states)

if __name__ == '__main__':
    print(f"Sample: {run()}")
