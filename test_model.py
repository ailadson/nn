from net import Net
from data_set_functions import *
import pickle
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_observations = list(zip(mnist.test.images, mnist.test.labels))

def load_model(name):
    return pickle.load( open( f"./models/{name}.p", "rb" ) )

nn = load_model("sig510_sig203")
loss, misclassification = evaluate(test_observations, nn)
print(f"Test Loss: {loss} | Test Misclassification: {misclassification}")
