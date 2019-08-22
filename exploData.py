import numpy as np
import cv2
import matplotlib.pyplot as plt

gimg = cv2.imread('Breast_cancer_cells.jpg', cv2.IMREAD_GRAYSCALE)
cimg = cv2.imread('Breast_cancer_cells.jpg', cv2.IMREAD_COLOR)
print(gimg.shape) 
print(cimg.shape)

plt.imshow(gimg)
plt.imshow(cimg)
plt.show()

