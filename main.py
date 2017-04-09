import config
from functions.data_prep import get_mnist_data
from functions.training import train_epoch
from net import Net
from trainer import Trainer

test_set, train_set, validation_set = get_mnist_data()

NUM_EPOCHS = 100
def run_training(nn, learning_rate, batch_size):
    config.start_profiler()
    try:
        trainer = Trainer(nn, learning_rate)
        for i in range(NUM_EPOCHS):
            train_epoch(
                nn, trainer, i, train_set, validation_set, batch_size
            )
    finally:
        config.stop_profiler()

# Trained to ~95% accuracy after 1 epoch.
def train_simple_fc_net():
    nn = Net()
    nn.add_rank3_input_layer((1, 28, 28))
    nn.add_flatten_layer()
    nn.add_fc_layer(100, "relu")
    nn.add_fc_layer(10, "relu")
    nn.add_softmax_layer()
    run_training(nn, 0.1, 10)

def train_one_layer_conv_net():
    nn = Net()
    nn.add_rank3_input_layer((1, 28, 28))
    nn.add_conv_layer(
        num_out_layers = 20,
        kernel_height = 3,
        kernel_width = 3,
        activation_fn_name = "relu"
    )
    nn.add_max_pool_layer()
    nn.add_flatten_layer()
    nn.add_fc_layer(100, "relu")
    nn.add_fc_layer(10, "relu")
    nn.add_softmax_layer()
    run_training(nn, 0.1, 10)

#train_simple_fc_net()
train_one_layer_conv_net()
