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
    for observation in observations:
        output = net.forward_propagate(observation[0])[0] #unwrapping
        loss_sum += cross_entropy(output, observation[1])
        guess = 1 if output >= 0.5 else 0
        if guess != observation[1]:
            misclassification += 1
    return (loss_sum/len(observations), misclassification/len(observations) )
