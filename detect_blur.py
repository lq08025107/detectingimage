# USAGE
# python detect_blur.py --images images

# import the necessary packages
from imutils import paths
import argparse
import cv2
import math
from matplotlib import pyplot as plt
import util
import numpy as np

# def variance_of_laplacian(image):
# 	# compute the Laplacian of the image and then return the focus
# 	# measure, which is simply the variance of the Laplacian
# 	return cv2.Laplacian(image, cv2.CV_64F).var()
#
# # loop over the input images
# for imagePath in paths.list_images(args["images"]):
# 	# load the image, convert it to grayscale, and compute the
# 	# focus measure of the image using the Variance of Laplacian
# 	# method
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	fm = variance_of_laplacian(gray)
# 	text = "Not Blurry"
#
# 	# if the focus measure is less than the supplied threshold,
# 	# then the image should be considered "blurry"
# 	if fm < args["threshold"]:
# 		text = "Blurry"
#
# 	# show the image
# 	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# 	cv2.imshow("Image", image)
# 	key = cv2.waitKey(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

def detect_blur(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	text = "Not Blurry"
	if fm < 100:
		text = "Blurry"
	cv2.putText(image, "{}:{:.2f}".format(text, fm),(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)

def detect_light_dark(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	avg_grayscale = cv2.mean(gray)[0]
	text = "Normal"
	if avg_grayscale < 50:
		text = "Dark"
	elif avg_grayscale > 220:
		text = "Light"
	cv2.putText(image, "{}:{:.2f}".format(text, avg_grayscale),(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("image", image)
	key = cv2.waitKey(0)

def detect_color_cast(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0], None,[16],[0, 180])
	hist = hist.flatten()
	ratio = max(hist) / sum(hist)
	print ratio
	text = "Normal"
	if ratio > 0.8:
		text = "Color Cast"

	cv2.putText(image, "{}:{:.2f}".format(text, ratio), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("image", image)

	key = cv2.waitKey(0)

def detect_no_signal(image):
	rows = image.shape[0]
	cols = image.shape[1]
	imageLt = image[0:(cols/2-1),0:(rows/2-1)]
	imageRt = image[cols/2:cols-1,0:(cols/2-1)]
	imageLb = image[0:(cols/2-1),rows/2:rows-1]
	imageRb = image[cols/2:cols-1,rows/2:rows-1]

	fHall = util.get_entropy_hist(image)
	fav1 = util.get_entropy_hist(imageLt)
	fav2 = util.get_entropy_hist(imageRt)
	fav3 = util.get_entropy_hist(imageLb)
	fav4 = util.get_entropy_hist(imageRb)

	fav = (fav1 + fav2 + fav3 + fav4) / 4
	fimgH = fHall + fav * 0.2
	text = "Normal"
	if fimgH < 1.0:
		text = "No Signal"

	cv2.putText(image, "{}:{:.2f}".format(text, fimgH), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("image", image)

	key = cv2.waitKey(0)

def detect_snow(image):
	#imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imagegray = image
	core = np.ones((5, 5), np.float)/25
	print core
	core_0 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
	core_45 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], np.float32)
	core_90 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], np.float32)
	core_135 = np.array([[1,1,0],[1,0,-1],[0,-1,-1]], np.float32)



	res = cv2.filter2D(imagegray, -1, core)
	res_0 = cv2.filter2D(imagegray, -1, core_0)
	res_45 = cv2.filter2D(imagegray, -1, core_45)
	res_90 = cv2.filter2D(imagegray, -1, core_90)
	res_135 = cv2.filter2D(imagegray, -1, core_135)

	cv2.imshow("conv", res)
	cv2.imshow("0 conv", res_0)
	cv2.imshow("45 conv", res_45)
	cv2.imshow("90 conv", res_90)
	cv2.imshow("135 conv", res_135)

	print res_0.shape
	print res_0
	cv2.waitKey(0)
if __name__ == "__main__":
	image = cv2.imread("images\\lena.jpg")
	#detect_blur(image)
	#detect_light_dark(image)
	#detect_color_cast(image)
	#detect_no_signal(image)
	detect_snow(image)
	#util.plot_rgb(image)

