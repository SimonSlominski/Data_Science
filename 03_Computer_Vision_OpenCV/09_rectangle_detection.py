import cv2
import imutils

# Read the image
image = cv2.imread('01_Document_Scanner/images/phone.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

# Add blur to image
gray_image = cv2.GaussianBlur(src=gray_image, ksize=(5, 5), sigmaX=0)

# Edges detection
edges = cv2.Canny(image=gray_image, threshold1=70, threshold2=200)

# Contours detection and sort them based on the contour area. Pick 10 larges contours.
contours = cv2.findContours(image=edges,
                            mode=cv2.RETR_TREE,
                            method=cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# The largest contour
cnt1 = cv2.drawContours(image=image.copy(),
                        contours=[contours[0]],
                        contourIdx=-1,
                        color=(0, 255, 0),
                        thickness=3)

# Rectangle detection
# screen_counter is a numpy array with 4 edges
screen_contour = None

for contour in contours:
    perimeter = cv2.arcLength(curve=contour, closed=True)
    approx = cv2.approxPolyDP(curve=contour, epsilon=0.015 * perimeter, closed=True)

    if len(approx) == 4:
        screen_contour = approx
        break

# Draw 4 points of every corner
vertices = cv2.drawContours(image=image.copy(),
                            contours=screen_contour,
                            contourIdx=-1,
                            color=(0, 255, 0),
                            thickness=3)

# Draw detected rectangle
screen_contour = cv2.drawContours(image=image.copy(),
                                  contours=[screen_contour],
                                  contourIdx=-1,
                                  color=(0, 255, 0),
                                  thickness=3)

cv2.imshow('img', screen_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
