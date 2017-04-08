from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary
from simple_trainer import SimpleTrainer
from trainer import Trainer
from data_set_functions import *
import time
import pickle
from tensorflow.examples.tutorials.mnist import input_data

def save_model(name, nn):
    pickle.dump( nn, open( f"models/{name}.p", "wb" ) )

def prompt_save(nn):
    ans = input("Save model? d - discard | anything else - save\n")
    if ans == "d":
        print("Model Disarded\n\n")
    else:
        name = input("Enter filename. Omit extension\n")
        save_model(name, nn)
        print(f"Model Saved As: ./models/{name}.p \n\n")

def load_model(name):
    return pickle.load( open( f"./models/{name}.p", "rb" ) )


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_observations = list(zip(reshape_images(mnist.test.images), mnist.test.labels))
validation_observations = list(zip(reshape_images(mnist.validation.images), mnist.validation.labels))
train_observations = list(zip(reshape_images(mnist.train.images), mnist.train.labels))

nn2 = Net()
nn2.add_rank3_input_layer((1, 28, 28))
nn2.add_conv_layer(3, 3, 20)
nn2.add_max_pool_layer()
nn2.add_conv_layer(3, 3, 20)
nn2.add_max_pool_layer()
nn2.add_flatten_layer()
# nn2.add_fc_layer(100)
nn2.add_output_layer(10)
#
t2 = Trainer(nn2, 1.0)
#
import cProfile
pr = cProfile.Profile()
pr.enable()

try:
    for i in range(100):
        train_epoch(nn2, t2, i, train_observations, validation_observations)
except BaseException as err:
    pr.disable()
    pr.dump_stats("stats.prof")
    raise err
    # prompt_save(nn2)

pr.disable()
pr.dump_stats("stats.prof")
