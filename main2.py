from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary
from simple_trainer import SimpleTrainer
from trainer import Trainer
from data_set_functions import *
import time
import pickle
from tensorflow.examples.tutorials.mnist import input_data


def reshape_images(images):
    return images.reshape([-1, 1, 28, 28])

def train_epoch(nn, t, epoch_num, training_set, validation_set):
    print(f"Epoch number {epoch_num}")
    batches = batch_data(training_set, 10)
    batch_d = None
    total_batch_loss = 0
    total_batch_misclassification = 0

    last_time = time.time()
    for i, batch in enumerate(batches):
        batch_d = t.train_with_examples(batch)
        # total_batch_loss += batch_d[0]
        # total_batch_misclassification += batch_d[1]
        if (i + 1) % 100 == 0:
            print(time.time() - last_time)
            print(f"Batch {i + 1} of {len(batches)}")
            last_time = time.time()

        if i > 500: return
            # loss, misclassification = evaluate(validation_set, nn)
            # print((loss, misclassification))

    # loss, misclassification = evaluate(validation_set, nn)

    # print(f"Training Loss: {total_batch_loss/len(batches)} | Training Misclassification: {total_batch_misclassification/len(batches)}")
    # print(f"Last Batch Loss: {batch_d[0]} | Last Batch Misclassification: {batch_d[1]}")
    print(f"Epoch number {epoch_num} Loss: {loss} | Epoch number {epoch_num} Misclassification: {misclassification}")

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
nn2.add_conv_layer(3,3,20)
nn2.add_max_pool_layer()
# nn2.add_conv_layer(5,5,40)
nn2.add_max_pool_layer()
nn2.add_flatten_layer()
nn2.add_fc_layer(100)
nn2.add_output_layer(10)
#
t2 = Trainer(nn2, 0.001)
#
import cProfile
pr = cProfile.Profile()
pr.enable()

try:
    for i in range(1):
        train_epoch(nn2, t2, i, train_observations, validation_observations)
except BaseException as err:
    pr.disable()
    pr.dump_stats("stats.prof")
    raise err
    # prompt_save(nn2)

pr.disable()
pr.dump_stats("stats.prof")
