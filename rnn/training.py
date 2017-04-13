import batcher
from config import *
import tensorflow as tf
import graph
from text_data_set import TextDataSet
from collections import namedtuple

RunConfig = namedtuple(
    'RunConfig', [
        'graph',
        'dataset',
        'train_batches',
        'val_batches',
        'saver',
        'initial_states'
    ]
)

def run_validation(session, epoch_idx, runConf):
    avg_loss = 0
    avg_classification_rate = 0
    states = runConf.initial_states
    g = runConf.graph
    print("Starting Validation")

    for (batch_x, batch_y) in runConf.val_batches:
        states, loss = session.run(
            [g.final_states, g.avg_loss],
            feed_dict={
                g.inputs: batch_x,
                g.outputs: batch_y,
                tuple(g.initial_states): tuple(states)
            }
        )

        avg_loss += loss
        # avg_classification_rate += class_rate

    avg_loss /= len(runConf.val_batches)
    # avg_classification_rate /= len(val_batches)
    runConf.saver.save(session, f"checkpoints/epoch_{epoch_idx}_l{avg_loss}")
    print(f"Epoch {epoch_idx} | Validation Loss: {avg_loss}")

def run_epoch(session, epoch_idx, runConf):
    states = runConf.initial_states
    g = runConf.graph

    for train_idx, (batch_x, batch_y) in enumerate(runConf.train_batches):
        states, loss, _ = session.run(
            [g.final_states, g.avg_loss, g.train_op],
            feed_dict={
                g.inputs: batch_x,
                g.outputs: batch_y,
                tuple(g.initial_states): tuple(states)
            }
        )
        if train_idx % PRINT_EPOCH_INFO_RATE == 0:
            print(f"Epoch {epoch_idx}, Batch {train_idx} | Loss: {loss}")

    run_validation(session, epoch_idx, runConf)


def run():
    dataset = TextDataSet(f'../datasets/{DATASET_FILENAME}')
    train_batches, val_batches = batcher.partition_batches(
        batcher.make_batches(
            dataset.text_as_one_hot,
            BATCH_SIZE,
            STEP_SIZE
        )
    )

    g = graph.build_graph(
        BATCH_SIZE, dataset.num_chars, NUM_LAYERS, STEP_SIZE, NUM_LSTM_UNITS
    )
    saver = tf.train.Saver()
    initial_states = graph.make_initial_states(
        BATCH_SIZE, NUM_LAYERS, NUM_LSTM_UNITS
    )

    runConf = RunConfig(
        graph=g,
        dataset=dataset,
        train_batches=train_batches,
        val_batches=val_batches,
        saver=saver,
        initial_states=initial_states
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch_idx in range(EPOCHS):
            run_epoch(session, epoch_idx, runConf)

run()
