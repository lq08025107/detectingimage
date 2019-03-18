import numpy as np
import matplotlib.pyplot as plt
import cv2

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
if __name__ == '__main__':

    vec = np.random.uniform(-1,1,size=(10)) #changed the size part
    print vec
    plt.bar(range(len(vec)), vec, linewidth=1)
    plt.show()
    print 'done'