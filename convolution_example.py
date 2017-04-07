from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter


def c2d(img, conv, mode):
    if mode != "valid":
        raise Exception("BAD STUFF!!")
    img_height, img_width = img.shape
    k_height, k_width = conv.shape
    out_height, out_width = img_height - k_height + 1, img_width - k_width + 1
    out = np.zeros([out_height, out_width], dtype=np.float32)

    for i in range(out_height):
        for j in range(out_width):
            patch = get_patch(img, i, j, k_height, k_width)
            out[i, j] = convolve(patch, conv)
    return out

def get_patch(img, i, j, k_height, k_width):
    return img[i:i+k_height, j:j+k_width]

def convolve(patch, conv):
    return (patch * conv).sum()


im = Image.open("./imgs/convolve_test2.jpg")
data = np.array(im)
print(data.shape)
data = data[:,:,1]

sobelx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

sobely = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
])

# sobely = np.zeros([3,3], dtype=np.float32)

outx = c2d(data, sobelx, mode="valid")
outy = c2d(data, sobely, mode="valid")
out = np.sqrt(outx**2 + outy**2)


out = np.absolute(out)
out = out / out.max()
out = out * 255
out = out.astype(np.uint8)
print(out)
# out = out*(out == maximum_filter(out,footprint=np.ones((2,2))))
# out = out * (out > 64)

img = Image.fromarray(out, 'L')
# img.show()
