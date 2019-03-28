#encoding=utf-8
import cv2
import math
import numpy as np
def detect_stripe(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", image)
    cv2.imshow("gray", image_gray)
    image_width = image_gray.shape[1]
    image_height = image_gray.shape[0]
    count = 0

    i = 3
    j = 0
    while i < image_height:
        j = 0
        while j < image_width-13:
            L1A = 0
            L2AB = 0
            #print math.fabs(int(image_gray[i][j]) - int(image_gray[i-3][j]))
            if math.fabs(int(image_gray[i][j]) - int(image_gray[i-3][j])) > 20:
                for h in range(0, 13):
                    L1A += math.fabs(int(image_gray[i][j+h]) - int(image_gray[i][j]))
                    L2AB +=math.fabs(int(image_gray[i-3][j+h]) - int(image_gray[i][j+h]))

                if L1A < 300 and L2AB > 100:
                    count += 1
                    j = j + 3
                    continue
            j = j + 1
        i = i + 20


    print count
    ratio = (count * 1.0) / (image_height * image_width / 20)
    print ratio


def detect_stripe1(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[0]
    print h.shape
    dfth = cv2.dft(np.float32(h))
    cv2.imshow(dfth)

if __name__ == '__main__':
    image = cv2.imread("images\\stripe.png")
    print image.shape
    detect_stripe(image)
    cv2.waitKey(0)