import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utility import segmentation_utils

image = cv.imread('Task.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

plt.figure(figsize=(10, 14))
plt.imshow(image_rgb)
plt.title("Изображение с координатами")

plt.xlabel("X")
plt.ylabel("Y")

plt.xticks(np.arange(0, image.shape[1], 100))
plt.yticks(np.arange(0, image.shape[0], 100))

plt.grid(False)
plt.show()

seeds = [
    (500, 450),  
    (500, 700),  
    (600, 400),
    (600, 700),
    (600, 520),
]

threshold = 15  

def plot_histogram(image_hsv):
    h, s, v = cv.split(image_hsv)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(h.ravel(), bins=256, color='orange', alpha=0.7, label='Оттенок')
    
    plt.hist(s.ravel(), bins=256, color='green', alpha=0.7, label='Насыщенность')
    
    plt.hist(v.ravel(), bins=256, color='blue', alpha=0.7, label='Яркость')
    
    plt.title('Гистограмма каналов HSV')
    plt.xlabel('Интенсивность пикселей')
    plt.ylabel('Частота')
    plt.legend()
    plt.show()

plot_histogram(image_hsv)

mask = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)
result = cv.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image_rgb)

plt.subplot(1, 2, 2)
plt.title("Выделенный верх одежды")
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))

plt.show()



