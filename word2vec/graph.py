from collections import namedtuple
import tensorflow as tf
import math

Graph = namedtuple("Graph",[
    "inputs", "labels", "loss_avg", "train_op", "embedding_matrix"
])

def build_graph(num_vocab_words, dim_embedding, num_neg_samples):
    ipt = tf.placeholder(tf.int32, [None])
    ctx_word = tf.placeholder(tf.int32, [None])

    embedding_mat = tf.Variable(
        tf.truncated_normal([num_vocab_words, dim_embedding]),
        name="embedding_matrix"
    )

    weights = tf.Variable(
        tf.truncated_normal(
            [num_vocab_words, dim_embedding],
            stddev=(1/math.sqrt(dim_embedding))
        ),
        name="weights"
    )

    biases = tf.Variable(
        tf.zeros([num_vocab_words]),
        name="biases"
    )

    embedded_ipts = tf.nn.embedding_lookup(embedding_mat, ipt)


    losses = tf.nn.sampled_softmax_loss(
        weights,
        biases,
        tf.reshape(ctx_word, [-1, 1]),
        embedded_ipts,
        num_neg_samples,
        num_vocab_words
    )

    loss_avg = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_avg)

    return Graph(ipt, ctx_word, loss_avg, train_op, embedding_mat)
