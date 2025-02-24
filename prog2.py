import cv2
import numpy as np
import sys

def detect_and_match_features(img_l, img_r, num_matches=50):
    sift = cv2.SIFT_create()
    kp_l, descriptors_l = sift.detectAndCompute(img_l, None)
    kp_r, descriptors_r = sift.detectAndCompute(img_r, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_l, descriptors_r)
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    return kp_l, kp_r, matches

def rectify(img_l, kp_l, img_r, kp_r, matches):
    pts_l = np.int32([kp_l[m.queryIdx].pt for m in matches])
    pts_r = np.int32([kp_r[m.trainIdx].pt for m in matches])
    F, inliers = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC)
    pts_l = pts_l[inliers.ravel() == 1]
    pts_r = pts_r[inliers.ravel() == 1]

    h_l, w_l = img_l.shape
    h_r, w_r = img_r.shape
    _, H_l, H_r = cv2.stereoRectifyUncalibrated(np.float32(pts_l), np.float32(pts_r), F, imgSize=(h_l, w_l))

    img_l_rect = cv2.warpPerspective(img_l, H_l, (w_l, h_l))
    img_r_rect = cv2.warpPerspective(img_r, H_r, (w_r, h_r))
    return img_l_rect, img_r_rect

def stereo_block_matching(img_l_rect, img_r_rect, window_size=5, max_disparity=100, lamb=1):
    h, w = img_l_rect.shape
    disparity_map = np.zeros((h, w), np.float32)
    costs = np.full((h, w, max_disparity), np.inf)
    kernel = np.ones((window_size, window_size), dtype=np.float32)

    left_sq = img_l_rect.astype(np.float32) ** 2
    left_sq_sum = cv2.filter2D(left_sq, -1, kernel)

    for d in range(0, max_disparity):
        shifted_r = np.roll(img_r_rect, -d, axis=1)
        shifted_r[:, -d:] = 0

        right_sq = shifted_r.astype(np.float32) ** 2
        right_sq_sum = cv2.filter2D(right_sq, -1, kernel)
        cross_term = cv2.filter2D(img_l_rect.astype(np.float32) * shifted_r, -1, kernel)

        ssd = left_sq_sum - 2*cross_term + right_sq_sum
        costs[:, :, d] = ssd

    disparity_range = np.arange(max_disparity).reshape(1, 1, -1)
    dp = np.zeros((h, w, max_disparity), dtype=np.float32)
    dp[:, 0, :] = costs[:, 0, :]

    for x in range(1, w):
        penalties = np.abs(disparity_range - disparity_range.transpose((0, 2, 1))) * lamb
        min_penalties = np.min(dp[:, x - 1, :, np.newaxis] + penalties, axis=1)
        dp[:, x, :] = costs[:, x, :] + min_penalties

    disparity_map = np.argmin(dp, axis=2)
    return disparity_map

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prog2.py <left_image_name> <right_image_name> <scalefactor> <output_disparity_name>")
        sys.exit(1)

    img_l = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    if type(img_l) == type(None):
        print("File not found:", sys.argv[1])
        sys.exit(1)
    if type(img_r) == type(None):
        print("File not found:", sys.argv[2])
        sys.exit(1)
    
    # kp_l, kp_r, matches = detect_and_match_features(img_l, img_r, num_matches=100)
    # img_matches = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite('matches.png', img_matches)

    # img_l_rect, img_r_rect = rectify(img_l, kp_l, img_r, kp_r, matches)
    # cv2.imwrite('img_l_rect.png', img_l_rect)
    # cv2.imwrite('img_r_rect.png', img_r_rect)

    disparities = stereo_block_matching(img_l, img_r, window_size=7, max_disparity=64, lamb=10)
    scaled = np.uint8(np.clip(disparities * int(sys.argv[3]), 0, 255))
    cv2.imwrite(sys.argv[4], scaled)

