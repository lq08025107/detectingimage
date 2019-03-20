import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def plot_rgb(image):
    # RGB
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = [cv2.calcHist([imgRGB], [k], None, [256], [0, 256]) for k in range(3)]
    x = np.arange(256) + 0.5

    plt.subplot(221)
    plt.imshow(imgRGB)
    plt.subplot(222)
    plt.bar(x, hist[0].flatten(), color='r', edgecolor='r')
    plt.subplot(223)
    plt.bar(x, hist[1].flatten(), color='g', edgecolor='g')
    plt.subplot(224)
    plt.bar(x, hist[2].flatten(), color='b', edgecolor='b')
    plt.show()

    # HSV
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = [cv2.calcHist([imgHSV], [0], None, [100], [0, 180]), \
            cv2.calcHist([imgHSV], [1], None, [100], [0, 256])]
    x = np.arange(100) + 0.5

    plt.subplot(211), plt.bar(x, hist[0].flatten())
    plt.subplot(212), plt.bar(x, hist[1].flatten())
    plt.show()

def get_entropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    image = np.array(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            val = image[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if tmp[i] == 0:
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

def get_entropy_hist(image):
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histgray = cv2.calcHist([imagegray], [0], None, [255], [0, 255]).flatten()
    tmp = []
    sum = 0
    result = 0
    for i in range(26):
        tmp.append(0)
    for i in range(25):
        for j in range(10):
            sum += histgray[i*10 + j]
        tmp[i] = sum / (imagegray.shape[0] * imagegray.shape[1])
        sum = 0
    for i in range(250, 255):
        sum += histgray[i]
    tmp[25] = sum / (imagegray.shape[0] * imagegray.shape[1])

    for i in range(26):
        if tmp[i] == 0:
            result= result
        else:
            result = result - tmp[i] * (math.log(tmp[i]) / math.log(2.0))
            #/ math.log(2.0)
    return result


# imagegray = image
# core = np.ones((5, 5), np.float) / 25
# print core
# core_0 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
# core_45 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], np.float32)
# core_90 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
# core_135 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32)
#
# res = cv2.filter2D(imagegray, -1, core)
# res_0 = cv2.filter2D(imagegray, -1, core_0)
# res_45 = cv2.filter2D(imagegray, -1, core_45)
# res_90 = cv2.filter2D(imagegray, -1, core_90)
# res_135 = cv2.filter2D(imagegray, -1, core_135)
#
# cv2.imshow("conv", res)
# cv2.imshow("0 conv", res_0)
# cv2.imshow("45 conv", res_45)
# cv2.imshow("90 conv", res_90)
# cv2.imshow("135 conv", res_135)
if __name__ == '__main__':

    image = cv2.imread("images\\nosignal.png")
    entropy = get_entropy_hist(image)
    print entropy

