import sys
import numpy as np
import cv2 as cv
import math
import imutils

def mask():
    mask = np.ones((7,7))
    mask[0, 0] = 0
    mask[0, 1] = 0
    mask[0, 5] = 0
    mask[0, 6] = 0
    mask[1, 0] = 0
    mask[1, 6] = 0
    mask[5, 0] = 0
    mask[5, 6] = 0
    mask[6, 0] = 0
    mask[6, 1] = 0
    mask[6, 5] = 0
    mask[6, 6] = 0
    return mask
	
def plot_image(image, title):
    plt.figure()

    plt.title(title)
    plt.imshow(image, cmap = 'gray')

    plt.show()
	
def SUSAN_corner_detector(img):
    img = img.astype(np.float64)
    g = 18.5
    circle = mask()
    corners = np.zeros(img.shape)

    for i in range(3, img.shape[0] - 3):
        for j in range(3, img.shape[1] - 3):
            im = np.array(img[i - 3:i + 4, j - 3:j + 4])
            im =  im[circle == 1]
            im0 = img[i, j]
            val = np.sum(np.exp(-((im - im0) / 10)**6))
            if val <= g:
                val = g - val
            else:
                val = 0
            corners[i, j] = val
    return corners

#Эта функция может считаться пару минут в зависимости от размера изображения	
def distance_from_positives(corners):
    distance = np.zeros((corners.shape[0], corners.shape[1]))
    for i in range(0, corners.shape[0]):
        for j in range(0, corners.shape[1]):
            if corners[i, j] > 0:
                distance[i, j] = 0
            else:
                radius = 0
                check = 0
                while radius < 10 and check == 0:
                    radius = radius + 1
                    for m in range(max(i - radius, 0), min(i + radius + 1, corners.shape[0])):
                        for n in range(max(j - radius, 0), min(j + radius + 1, corners.shape[1])):
                            if corners[m, n] > 0:
                                check = check + 1
                distance[i, j] = radius
    return distance
	
def count_objects(img):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edgedImage = cv.Canny(grayImage, 50, 130)
    imgContours = cv.findContours(edgedImage.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    grabbedContours = imutils.grab_contours(imgContours)
    return len(grabbedContours)
	
img = cv.imread("graph_1.png", 0)
vertexNumber = int(input())

corners = SUSAN_corner_detector(img)
distance = distance_from_positives(corners)

img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img[distance <= 8] = [0, 0, 0]
img[distance > 8] = [255, 255, 255]

print ("Number of edge intersections:", count_objects(img) - vertexNumber)
