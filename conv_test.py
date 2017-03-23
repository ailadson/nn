from PIL import Image
import math
from net import Net
import numpy as np
from functions import deconvolve2d
from simple_trainer import SimpleTrainer

# import data
im = Image.open("./imgs/convolve_test3.jpeg")
data = np.array(im)
data = np.transpose(data, (2,0,1))
data = data[0:1, :, :]

# create net
net = Net()
net.add_rank3_input_layer(data.shape)
net.add_conv_layer(3, 3, 1)
net.add_output_layer()

# get observed output
net.layers[1].weights[0, 0] = np.array([
    [-10, 0, 0],
    [0, 0, 0],
    [0, 0, 10]
])

data = data.astype('float64')
observed = np.copy(net.forward_propagate(data))

# randomize weights
net.layers[1].weights = net.layers[1].initialize_weights()

t = SimpleTrainer(net, 1.0)
prev_cost = 10 ** 10
for i in range(10000):
    t.train_with_examples([(data, observed)])
    out = net.forward_propagate(data)
    cost = math.sqrt(((observed - out) ** 2).sum())
    if cost > prev_cost:
        print("COST INCREASED")
        t.learning_rate *= 0.1
    prev_cost = cost


print(net.layers[1].weights)
print(net.forward_propagate(data))
print(cost)
