import cv2
import numpy as np
from PIL import Image

image = cv2.imread('lab4_image.png', cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold_image = cv2.threshold(gray_image, 194, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(threshold_image, kernel, iterations=1)
cv2.imwrite('threshold_image.png', eroded_image)

Image.open('threshold_image.png').show()
