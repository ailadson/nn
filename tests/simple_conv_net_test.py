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
data = data.astype('float32')

# create net
net = Net()
net.add_rank3_input_layer(data.shape)
net.add_conv_layer(3, 3, 1)
net.layers[1].weights[0, 0] = np.array([
    [0, 0, -10],
    [0, 0, 0],
    [10, 0, 0]
])
# net.add_max_pool_layer()
net.add_conv_layer(3, 3, 1)
net.layers[2].weights[0, 0] = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
# net.add_max_pool_layer()
net.add_rank3output_layer()

# get observed output
observed = np.copy(net.forward_propagate(data))
# img = Image.fromarray(observed[0,:,:].astype("uint8"), 'L')
# img.show()

# randomize weights
net.layers[1].weights += np.random.uniform(size=[1, 1, 3, 3]).astype(np.float32) #net.layers[1].initialize_weights()
net.layers[2].weights += np.random.uniform(size=[1, 1, 3, 3]).astype(np.float32) #= net.layers[2].initialize_weights()
net.layers[2].deriv_wrt_weights = net.layers[2].regularized_deriv_wrt_weights
# import types
# def my_has_weights(self):
#     return False
# net.layers[2].has_weights = types.MethodType(my_has_weights, net.layers[2])

t = SimpleTrainer(net, 0.01)
prev_cost = 10 ** 10
for i in range(100000):
    if i % 1000 == 0:
        print(i)
        print(net.layers[1].weights)
        print(net.layers[2].weights)
    t.train_with_examples([(data, observed)])
    out = net.forward_propagate(data)
    cost = math.sqrt(((observed - out) ** 2).sum())
    if cost > prev_cost:
        print("COST INCREASED")
        t.learning_rate *= 0.1
    prev_cost = cost


print("~Weights~")
print(net.layers[1].weights)
# print(observed)
# print(cost)
