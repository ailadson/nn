import config
from net import Net
from file_reader import FileReader
from vocabulary import Vocabulary
from simple_trainer import SimpleTrainer
from trainer import Trainer
from data_set_functions import *
import time
import pickle

test_observations, validation_observations,train_observations = get_mnist_data()

# Build Net
nn = Net()
nn.add_rank3_input_layer((1, 28, 28))
nn.add_conv_layer(3, 3, 20)
nn.add_max_pool_layer()
nn.add_conv_layer(3, 3, 20)
nn.add_max_pool_layer()
nn.add_flatten_layer()
# nn.add_fc_layer(100)
nn.add_output_layer(10)

# Train
config.start_profiler()
try:
    t2 = Trainer(nn, 1.0)
    for i in range(100):
        train_epoch(nn, t2, i, train_observations, validation_observations)
finally:
    config.stop_profiler()
