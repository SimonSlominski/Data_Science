"""
Detection problems may occur if the background of the image is bright
"""

from numpy.linalg import norm
from skimage.filters import threshold_local
import numpy as np
import imutils
import cv2

image = cv2.imread('images/paragon_1.jpg')

# Image size standardization
# Keep a copy of the original image for later transformations
original_image = image.copy()

# Keep the original image's aspect ratio
ratio = image.shape[0] / 500.0

# Resize up to 500 px. From (600, 450, 3) to (500m 375, 3)
image = imutils.resize(image, height=500)

# Image conversion to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Edge detection
edges = cv2.Canny(gray_image, threshold1=75, threshold2=200)

# Add blur
gray_image = cv2.GaussianBlur(gray_image, ksize=(5, 5), sigmaX=0)

# Find contours
contours = cv2.findContours(image=edges.copy(),
                            mode=cv2.RETR_LIST,
                            method=cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Finding the document outline
screen_contour = None
for contour in contours:
    # calculate the perimeter of each figure found
    perimeter = cv2.arcLength(curve=contour, closed=True)

    # approximation of the rectangle curve
    approx = cv2.approxPolyDP(curve=contour, epsilon=0.02 * perimeter, closed=True)

    if len(approx) == 4:
        screen_contour = approx
        break

# Display found vertices
# vertices = cv2.drawContours(image, contours=screen_contour, contourIdx=-1, color=(0, 255, 0), thickness=10)

# Extraction of vertices
points = screen_contour.reshape(4, 2)
points = points * ratio

# Create empty numpy array
rectangle = np.zeros((4, 2), dtype='float32')

total = points.sum(axis=1)
rectangle[0] = points[np.argmin(total)]
rectangle[2] = points[np.argmax(total)]

difference = np.diff(points, axis=1)
rectangle[1] = points[np.argmin(difference)]
rectangle[3] = points[np.argmax(difference)]


a, b, c, d = rectangle

width1 = norm(c - d)
width2 = norm(b - a)
max_width = max(int(width1), int(width2))

height1 = norm(b - c)
height2 = norm(a - d)
max_height = max(int(height1), int(height2))

vertices = np.array([
                    [0, 0],
                    [max_width -1, 0],
                    [max_width -1, max_height - 1],
                    [0, max_height - 1]
                    ], dtype='float32')

# Transformation matrix 3x3
M = cv2.getPerspectiveTransform(rectangle, vertices)

# Transfer of document to image
out = cv2.warpPerspective(src=original_image, M=M, dsize=(max_width, max_height))

# To grayscale
out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

# Calculation of the threshold mask based on the proximity of pixels
T = threshold_local(image=out, block_size=11, offset=10, method='gaussian')
out = (out > T).astype('uint8') * 255


cv2.imshow('img', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
