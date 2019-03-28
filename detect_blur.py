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
	imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imagemedian = cv2.medianBlur(imagegray, 3)
	#imagemebil = cv2.bilateralFilter(imagegray, 9,75,75)
	#cv2.imshow("bil image", imagemebil)
	noise = imagegray - imagemedian
	sum = 0
	noise = noise.flatten()
	for i in range(len(noise)):
		if noise[i] > 10:
			sum += 1

	ratio = sum*1.0 / (image.shape[0] * image.shape[1])
	text = "Normal"
	if ratio > 0.7:
		text = "Snow"

	cv2.putText(image, "{}:{:.2f}".format(text, ratio), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)

def detect_noise(image):
	howmany = 0
	count = 0
	L1 = 0
	L2 = 0
	image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image_height = image.shape[0]
	image_width = image.shape[1]
	i = 1
	j = 1
	while i < image_height - 1:
		j = 1
		while j < image_width - 3:

			dataA = image_gray[i][j]
			dataB = image_gray[i][j + 2]
			if((i * j) % 1000 == 0):
				print str(int(dataA)) + " " + str(int(dataB))
				print math.fabs(int(dataA) - int(dataB))
				print "================================"
				howmany = howmany + 1
				#ath.fabs(int(dataA-dataB))
			if math.fabs(int(dataA)-int(dataB)) > 10:

				for k in range(-1, 2):
					for t in range(-1, 2):
						tempA = math.fabs(int(image_gray[i+k][j+t]) - int(dataA))
						tempB = math.fabs(int(image_gray[i+k][j+2+t])- int(dataB))
						L1 = L1 + tempA
						L2 = L2 + tempB

				if (L1 > L2) and (L1 > 10):
					count = count + 1
					if j < image_width -3:
						j = j + 3
				elif (L1 <= L2) and (L2 > 10):
					count = count + 1
					if( j < image_width -3):
						j = j + 3
			j = j + 1
			L1 = 0
			L2 = 0
		i = i + 1


	print howmany
	ratio = count*1.0 / (image_width * image_height)
	print ratio





if __name__ == "__main__":
	image = cv2.imread("images\\guobo.jpg")
	#detect_blur(image)
	#detect_light_dark(image)
	#detect_color_cast(image)
	#detect_no_signal(image)
	#detect_snow(image)
	detect_noise(image)
	#util.plot_rgb(image)
