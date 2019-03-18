# USAGE
# python detect_blur.py --images images

# import the necessary packages
from imutils import paths
import argparse
import cv2
import math
from matplotlib import pyplot as plt
import util

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
	hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
	hist = hist.flatten()
	ratio = max(hist) / sum(hist)
	print ratio
	text = "Normal"
	if ratio > 0.2:
		text = "Color Cast"

	cv2.putText(image, "{}:{:.2f}".format(text, ratio), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("image", image)

	key = cv2.waitKey(0)



if __name__ == "__main__":
	image = cv2.imread("images\\yellow2.jpg")
	#detect_blur(image)
	#detect_light_dark(image)
	detect_color_cast(image)
	#util.plot_rgb(image)

