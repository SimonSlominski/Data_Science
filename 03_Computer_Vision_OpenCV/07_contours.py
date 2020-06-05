import cv2

original_img = cv2.imread('images/python.png')
img = original_img.copy()

# Img to grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Add mask
thresh = cv2.threshold(src=gray, thresh=250, maxval=255, type=cv2.THRESH_BINARY)[1]

# Detect contours
contours = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[0]
print(f'[INFO] Number of contours: {len(contours)}')

# By changing the index contours = [contours [1]] we will get the next contours of the image
img_cnt = cv2.drawContours(image=img.copy(),
                           contours=[contours[1]],
                           contourIdx=-1,
                           color=(0, 255, 0),
                           thickness=2)

# Detect the bigger contours
max_area = 0
for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour=contour, oriented=True)
    if area > max_area:
        max_area = area
        idx_flag_area = idx

print(f'[INFO] Contour index with the largest area: {idx_flag_area}, Area: {max_area}')

# Show biggest contours
img_cnt_max_area = cv2.drawContours(image=img.copy(),
                                   contours=[contours[idx_flag_area]],
                                   contourIdx=-1,
                                   color=(0, 255, 0),
                                   thickness=2)
cv2.imshow('max area', img_cnt_max_area)
cv2.waitKey(0)
