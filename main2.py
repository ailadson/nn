from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary
from trainer import Trainer
from data_set_functions import *
import time
import pickle
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
test_observations = list(zip(mnist.test.images, mnist.test.labels))
validation_observations = list(zip(mnist.validation.images, mnist.validation.labels))
train_observations = list(zip(mnist.train.images, mnist.train.labels))


def train_epoch(nn, t, epoch_num, training_set, validation_set):
    start_time = time.time()
    print(f"Epoch number {epoch_num}")
    batches = batch_data(training_set, 1)
    batch_d = None
    batch_loss = 0
    batch_misclas_rate = 0

    for i, batch in enumerate(batches):
        batch_d = t.train_with_examples(batch)
        batch_loss += batch_d[0]
        batch_misclas_rate+= batch_d[1]
        if (i+1) % 11000 == 0:
            print(f"Batch {i + 1} of {len(batches)}")

    loss, misclassification = evaluate(validation_set, nn)

    print(f"Avg Batch Loss: {batch_loss/len(batches)} | Avg Batch Misclassification: {batch_misclas_rate/len(batches)}")
    print(f"Last Batch Loss: {batch_d[0]} | Last Batch Misclassification: {batch_d[1]}")
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

print("MODEL 2")
print("~"*20)
#Epoch number 9 Loss: 1.4840154230518507 | Epoch number 9 Misclassification: 0.0242
nn2 = Net(mnist.test.images.shape[1])
nn2.add_layer(510)
nn2.add_layer(203)
nn2.add_output_layer(10)

t2 = Trainer(nn2, 0.1)

for i in range(10):
    train_epoch(nn2, t2, i, train_observations, validation_observations)
prompt_save(nn2)
