import config
import random
from functions import *
from tensorflow.examples.tutorials.mnist import input_data

def batch_data(observations, num_size):
    random.shuffle(observations)
    batches = []

    start = 0
    while start < len(observations):
        batches.append(observations[start:start + num_size])
        start += num_size

    return batches

def evaluate(observations, net):
    loss_sum = 0
    misclassification = 0
    num_observations = len(observations)

    for (ipt, expected_output) in observations:
        true_class_index = list(expected_output).index(1)
        output = net.forward_propagate(ipt)
        loss_sum += cross_entropy(output, true_class_index)
        estimated_true_class_idx = list(output).index(max(output))
        misclassification += (
            1 if true_class_index != estimated_true_class_idx else 0
        )

    avg_loss = loss_sum / num_observations
    misclassification_rate = misclassification / num_observations
    return (avg_loss, misclassification_rate)

def get_mnist_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    test = list(zip(reshape_images(mnist.test.images), mnist.test.labels))
    validation = list(zip(reshape_images(mnist.validation.images), mnist.validation.labels))
    train = list(zip(reshape_images(mnist.train.images), mnist.train.labels))

    return (test, validation, training)

def reshape_images(images):
    return images.reshape([-1, 1, 28, 28])

def segment_data(observations):
    random.shuffle(observations)
    training_end = int(len(observations) * 0.8)
    validation_end = training_end + int(len(observations) * .1)
    return (
        observations[:training_end],
        observations[training_end:validation_end],
        observations[validation_end:]
    )

def train_epoch(nn, trainer, epoch_num, training_set, validation_set):
    print(f"Epoch number {epoch_num}")
    batches = batch_data(training_set, config.BATCH_SIZE)

    for batch_idx, batch in enumerate(batches):
        t.train_with_examples(batch)
        if (batch_idx + 1) % config.BATCHES_PER_EVALUATION == 0:
            loss, misclassification = evaluate(validation_set, nn)
            print(f"Epoch number {epoch_num} | "
                  f"Batch {batch_idx + 1}/{len(batches)} | "
                  f"Loss: {loss} | "
                  f"Misclassification: {misclassification}")

    loss, misclassification = evaluate(validation_set, nn)
    print(f"Epoch number {epoch_num} | Loss: {loss} | "
          f"Misclassification: {misclassification}")
