import random
from tensorflow.examples.tutorials.mnist import input_data

def batch_data(observations, num_size):
    random.shuffle(observations)
    batches = []

    start = 0
    while start < len(observations):
        batches.append(observations[start:start + num_size])
        start += num_size

    return batches

def get_mnist_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    test = list(zip(
        reshape_images(mnist.test.images), mnist.test.labels
    ))
    train = list(zip(
        reshape_images(mnist.train.images), mnist.train.labels
    ))
    validation = list(zip(
        reshape_images(mnist.validation.images), mnist.validation.labels
    ))

    return (test, train, validation)

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
