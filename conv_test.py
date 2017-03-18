from PIL import Image
from conv_layer import ConvolutionalLayer
from rank3input_layer import Rank3Input
import numpy as np

im = Image.open("./imgs/convolve_test3.jpeg")
data = np.array(im)
data = np.transpose(data, (2,0,1))
inputlayer = Rank3Input(data.shape)
inputlayer.set_input(data)
convlayer = ConvolutionalLayer(inputlayer, 3, 3, 3)
convlayer.weights = np.array([
    [
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        np.zeros([3, 3]),
        np.zeros([3, 3])
    ],
    [
        np.zeros([3, 3]),
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        np.zeros([3, 3])
    ],
    [
        np.zeros([3, 3]),
        np.zeros([3, 3]),
        [[-1,-2,-1],[0,0,0],[1,2,1]],
    ],
])
print(data.shape)
print(convlayer.output.shape)
convlayer.forward_propagate()
print(convlayer.output)
out = np.transpose(convlayer.output, (1,2,0))
# out = np.absolute(out)
# out = out / out.max()
# out = out * 255
out = out.astype(np.uint8)

img = Image.fromarray(out, 'RGB')
img.show()


from PIL import ImageFile

ImageFile.MAXBLOCK = 2**20

img.save("relu.jpg", "JPEG", quality=80, optimize=True, progressive=True)
