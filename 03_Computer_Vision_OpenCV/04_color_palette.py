import cv2
import numpy as np

def nothing(x):
    pass

img = np.zeros(shape=(300, 500, 3), dtype='uint8')
cv2.namedWindow('box')

cv2.createTrackbar('Red', 'box', 0, 255, nothing)
cv2.createTrackbar('Green', 'box', 0, 255, nothing)
cv2.createTrackbar('Blue', 'box', 0, 255, nothing)

while True:
    cv2.imshow('box', img)

    r = cv2.getTrackbarPos('Red', 'box')
    g = cv2.getTrackbarPos('Green', 'box')
    b = cv2.getTrackbarPos('Blue', 'box')

    img[:] = [b, g, r]

    if cv2.waitKey(20) == 27:
        break
