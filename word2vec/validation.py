import numpy as np
import tensorflow as tf

class Validator():
    def __init__(self, embedding_mat, batcher):
        self.batcher = batcher
        self.embedding_mat = embedding_mat
        self.example_words = [ i for i in range(1000, 1032) ]
        self.sim_scores_ = None

    def sim_scores(self):
        # if self.sim_scores_ is not None:
        #     return self.sim_scores_

        examples = tf.constant(self.example_words)
        norm_embedding_mat = tf.nn.l2_normalize(self.embedding_mat, 0)
        embedded_examples = tf.nn.embedding_lookup(norm_embedding_mat, examples)

        self.sim_scores_ = tf.matmul(
            embedded_examples,
            tf.transpose(norm_embedding_mat)
        )

        return self.sim_scores_

    def run(self, run_info):
        session = run_info.session
        scores = session.run(self.sim_scores())
        for i, exp_word in enumerate(self.example_words):
            score_row = np.argsort(scores[i,:])
            words = map(
                lambda word_int: self.batcher.int_to_word(word_int),
                score_row[-16:]
            )
            print(f"Top 16 Words Similar to '{self.batcher.int_to_word(exp_word)}''")
            print(list(words))
            print("~"*15)
