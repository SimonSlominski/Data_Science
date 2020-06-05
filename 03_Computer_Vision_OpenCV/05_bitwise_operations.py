import cv2
import imutils

# Load imgages and change the logo shape
img = cv2.imread('images/view.jpg')
logo = cv2.imread('images/python.png')
logo = imutils.resize(logo, height=150)

# Show both images
# cv2.imshow('img', img)
# cv2.imshow('logo', logo)

# Cut Region of Interest
rows, cols, channels = logo.shape
roi = img[:rows, :cols]
# cv2.imshow('roi', roi)
# cv2.waitKey(0)

# Convert logo to grayscale
logo_gray = cv2.cvtColor(src=logo, code=cv2.COLOR_BGR2GRAY)

# Create mask for logo imgage
mask = cv2.threshold(src=logo_gray, thresh=220, maxval=255, type=cv2.THRESH_BINARY)[1]

# Invert mask
mask_inv = cv2.bitwise_not(mask)

# Connecting two images
img_bg = cv2.bitwise_and(src1=roi, src2=roi, mask=mask)
logo_fg = cv2.bitwise_and(src1=logo, src2=logo, mask=mask_inv)

dst = cv2.add(img_bg, logo_fg)
img[:rows, :cols] = dst
# cv2.imshow('dst img', dst)
# cv2.waitKey(0)

# Show connected images
cv2.imshow('new_view', img)
cv2.waitKey(0)


