import config
from net import Net
from trainer import Trainer
from functions.training import get_mnist_data, train_epoch
import time
import pickle

test_set, train_set, validation_set = get_mnist_data()

# Build Net
nn = Net()
nn.add_rank3_input_layer((1, 28, 28))
nn.add_conv_layer(
    num_out_layers = 20,
    kernel_height = 3,
    kernel_width = 3,
    activation_fn_name = "relu"
)
nn.add_max_pool_layer()
nn.add_conv_layer(
    num_out_layers = 40,
    kernel_height = 3,
    kernel_width = 3,
    activation_fn_name = "relu"
)
nn.add_max_pool_layer()
nn.add_flatten_layer()
nn.add_fc_layer(100, "relu")
nn.add_fc_layer(10, "relu")
nn.add_softmax_layer()

# Train
config.start_profiler()
try:
    trainer = Trainer(nn, 1.0)
    for i in range(100):
        train_epoch(nn, trainer, i, train_set, validation_set)
finally:
    config.stop_profiler()
