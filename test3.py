import cv2
import numpy as np

# Load images in grayscale
img_l = cv2.imread('data/cones_l.png', cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread('data/cones_r.png', cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoSGBM_create(minDisparity=0,
                                numDisparities=64,  # Disparity range
                                blockSize=5,        # Block size
                                P1=8 * 3 * 5 ** 2,  # Penalty for disparity change in small window
                                P2=32 * 3 * 5 ** 2, # Penalty for disparity change in large window
                                disp12MaxDiff=1,
                                preFilterCap=63,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=32)

disparity = stereo.compute(img_l, img_r).astype(np.float32) / 16.0
cv2.imwrite("disparity_map.png", disparity)