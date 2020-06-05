import cv2
import imutils

img = cv2.imread('images/view.jpg')
canny = cv2.Canny(image=img, threshold1=250, threshold2=250)
cv2.imshow('canny', canny)
cv2.waitKey(0)