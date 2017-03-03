import random
from functions import *

def segment_data(observations):
    random.shuffle(observations)
    training_end = int(len(observations) * 0.8)
    validation_end = training_end + int(len(observations) * .1)
    return( observations[:training_end], observations[training_end:validation_end], observations[validation_end:] )

def batch_data(observations, num_size):
    random.shuffle(observations)
    batches = []
    start = 0

    while start < len(observations):
        batches.append( observations[start : start + num_size] )
        start += num_size

    return batches


def evaluate (observations, net):
    loss_sum = 0
    misclassification = 0
    observations_length = len(list(observations))
    
    for observation in observations:
        observation_l = list(observation[1])
        output = net.forward_propagate(observation[0])
        loss_sum += cross_entropy(output, observation_l.index(1))
        estimated_true_class_idx = list(output).index( max(output) )
        misclassification += (1 if observation_l.index( 1 ) != estimated_true_class_idx else 0)
    return (loss_sum/observations_length, misclassification/observations_length )
