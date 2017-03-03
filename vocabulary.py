from file_reader import FileReader
from collections import Counter
import math
import numpy as np

class Vocabulary:
    SMOOTHING = 30
    LOR_THRESHOLD = math.log(1.15)

    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels
        self._feature_dict_ = None
        self._unfeature_dict_ = None

    def featurize_labels(self):
        return map(lambda val: [1,0] if val == 1 else [0,1], self.labels)

    def num_of_items(self):
        return len(self.feature_dict())

    def count_word_freq(self):
        counter = Counter()
        pos_counter = Counter()
        neg_counter = Counter()

        for review, label in zip(self.reviews, self.labels):
            for word in set(review):
                    counter[word] += 1
                    if label == 1:
                        pos_counter[word] += 1
                    else:
                        neg_counter[word] += 1

        return (counter, pos_counter, neg_counter)

    def review_odds_ratios(self):
        counter, pos_counter, neg_counter = self.count_word_freq()
        ratios = {}
        pos_reviews = sum(self.labels)
        neg_reviews = len(self.labels) - sum(self.labels)

        for word, freq in counter.items():
            pos_word_count = pos_counter[word]
            neg_word_count = neg_counter[word]
            ratios[word] = self.odds_ratio(pos_word_count, pos_reviews, neg_word_count, neg_reviews)

        return ratios

    def odds_ratio(self, pos_word_count, pos_reviews, neg_word_count, neg_reviews):
        pos = (pos_word_count + Vocabulary.SMOOTHING)/(pos_reviews + Vocabulary.SMOOTHING)
        neg = (neg_word_count + Vocabulary.SMOOTHING)/(neg_reviews + Vocabulary.SMOOTHING)
        return math.log(pos/neg)

    def rank_review_words(self):
        ratios = self.review_odds_ratios()
        ratio_list = list(ratios.items())
        ratio_list.sort(key=lambda pair: pair[1])
        return(ratio_list[:10], ratio_list[-10:])

    def filter_ratios(self):
        ratios = self.review_odds_ratios()
        filter_func = lambda pair: abs(pair[1]) > Vocabulary.LOR_THRESHOLD
        return list(filter(filter_func, ratios.items()))

    def feature_dict(self):
        if self._feature_dict_ is None:
            ratios = self.filter_ratios()
            self._feature_dict_ = { pair[0] : i for i, pair in enumerate(ratios) }
        return self._feature_dict_

    def unfeature_dict(self):
        if self._unfeature_dict_ is None:
            self._unfeature_dict_ = { v : k for k, v in self.feature_dict().items() }
        return self._unfeature_dict_

    def unfeature_word_indices(self, word_indices):
        unfeature_dict = self.unfeature_dict()
        words = []

        for idx in word_indices:
                words.append( unfeature_dict[idx] )

        return words

    def unfeature_review(self, review):
        unfeature_dict = self.unfeature_dict()
        words = []

        for i, word in enumerate(review):
            if word == 1:
                words.append( unfeature_dict[i] )

        return words

    def featurize_reviews(self):
        feature_dict = self.feature_dict()
        review_features = []

        for review in self.reviews:
            review_features.append( self.featurize_review(review, feature_dict) )

        return review_features

    def featurize_review(self, review, feature_hash):
        features = [0] * len(feature_hash)
        for word in review:
            if word in feature_hash:
                features[ feature_hash[word] ] = 1
        return np.array(features)
