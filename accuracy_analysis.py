import cv2
import numpy as np
import sys

output = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

print('Accuracy:', round(cv2.mean((output - gt) / 255)[0], 2))