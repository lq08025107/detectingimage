import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def magnitude(x, y):
    x_m = x * x
    y_m = y * y
    z_m = x_m + y_m
    return np.sqrt(z_m)
img = cv2.imread("images\\stripe1.png", 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
#magnitude_spectrum2 = 20 * np.log10(magnitude(dft_shift[:, :, 0], dft_shift[:,:,1]))

# sum = 0
# for i in range(magnitude_spectrum2.shape[0]):
#     for j in range(magnitude_spectrum2.shape[1]):
#         sum += magnitude_spectrum2[i][j]
# average = (sum*1.0) / (magnitude_spectrum2.shape[0]*magnitude_spectrum2.shape[1])
# print average

plt.subplot(121), plt.imshow(img, "gray")
plt.title("Input image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dft_shift, cmap="gray")
plt.title("DFT image"), plt.xticks([]), plt.yticks([])
plt.show()