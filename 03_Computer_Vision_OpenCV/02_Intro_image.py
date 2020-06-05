import cv2
import numpy as np

# Import and make a copy of original image
org_img = cv2.imread(filename='images/python.png')
img = org_img.copy()

# Show image
# cv2.imshow(winname='logo', mat=img)
# cv2.waitKey(delay=0)

#
height, width = img.shape[:2]
print(f'Height: {height}')
print(f'Width: {width}')

# Line
# cv2.line(img=img, pt1=(0, 0), pt2=(width, height), color=(0, 255, 0), thickness=5)
# cv2.imshow(winname='logo', mat=img)
# cv2.waitKey(0)

# Rectangle
# img = org_img.copy()
# cv2.rectangle(img=img, pt1=(200, 50), pt2=(400, 230), color=(255, 0, 0), thickness=3)
# cv2.imshow('logo', img)
# cv2.waitKey(0)

# Circle
# img = org_img.copy()
# cv2.circle(img=img, center=(300, 140), radius=90, color=(0, 0, 255), thickness=3)
# cv2.imshow('logo', img)
# cv2.waitKey(0)

# Polygon
# img = org_img.copy()
# pts = np.array([[300, 140],
#                 [200, 200],
#                 [200, 50],
#                 [300, 50]], dtype='int32').reshape((-1, 1, 2))
# cv2.polylines(img=img, pts=[pts], isClosed=False, color=(0, 255, 0), thickness=3)
# cv2.imshow('logo', img)
# cv2.waitKey(0)

# Text
# img = org_img.copy()
# cv2.putText(
#     img=img,
#     text='Python logo',
#     org=(20, 40),
#     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#     fontScale=1.5,
#     color=(0, 255, 0),
#     thickness=2
# )
# cv2.imshow('logo', img)
# cv2.waitKey(0)
