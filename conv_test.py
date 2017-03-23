from PIL import Image
from net import Net
import numpy as np
from functions import deconvolve2d
from simple_trainer import SimpleTrainer

# import data
im = Image.open("./imgs/convolve_test3.jpeg")
data = np.array(im)
data = np.transpose(data, (2,0,1))

# create net
net = Net()
net.add_rank3_input_layer((1, 4, 4))
net.add_conv_layer(3, 3, 1)
net.add_output_layer()

# get observed output
# net.layers[1].weights = np.array([
#     [
#         np.zeros([3, 3]),
#         [
#         [-1,0,1],
#         [-2,0,2],
#         [-1,0,1]
#         ],
#         np.zeros([3, 3])
#     ]
# ])
#
# data = data.astype('float64')
# data /= data.max()
# observed = np.copy(net.forward_propagate(data))
# observed /= observed.max()

# randomize weights
# net.layers[1].weights = net.layers[1].initialize_weights()

# train
data = np.ones([1, 4, 4]) * 4
observed = np.ones([1, 4, 4]) * 2

net.layers[1].weights.fill(1)
print(net.layers[1].weights)
t = SimpleTrainer(net, 0.0001)
for i in range(1000):
    t.train_with_examples([(data, observed)])
t.learning_rate *= 0.1
for i in range(9000):
    t.train_with_examples([(data, observed)])

print(net.layers[1].weights)
print(net.layers[1].weights.sum())
print(net.forward_propagate(data))

# print(data.shape)
# print(convlayer.output.shape)
# convlayer.forward_propagate()
# print(convlayer.output)
# out = np.transpose(convlayer.output, (1,2,0))
# # out = np.absolute(out)
# # out = out / out.max()
# # out = out * 255
# out = out.astype(np.uint8)
#
# img = Image.fromarray(out, 'RGB')
# img.show()
#
#
# from PIL import ImageFile
#
# ImageFile.MAXBLOCK = 2**20

# img.save("relu.jpg", "JPEG", quality=80, optimize=True, progressive=True)
