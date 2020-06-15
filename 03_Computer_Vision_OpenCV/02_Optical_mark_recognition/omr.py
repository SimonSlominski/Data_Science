from imutils import contours
import imutils
import numpy as np
import cv2

ANSWER_KEY = {0: 1,
              1: 3,
              2: 0,
              3: 2,
              4: 1,
              5: 3,
              6: 4,
              7: 1,
              8: 3,
              9: 0}

image = cv2.imread('images/answer_2.png')

# Convert to grayscale
gray = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)

# Blur the image
blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

# Edge detection
thresh = cv2.Canny(image=blurred, threshold1=70, threshold2=70)

# Contour extraction
cnts = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Extraction of relevant contours
question_contours = []

for contour in cnts:
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)

    if w >= 50 and h >= 50 and ar > 0.9 and ar < 1.1:
        question_contours.append(contour)

print(f"No. of response fields: {len(question_contours)}")

cnt_img = image.copy()
for contour in question_contours:
    cv2.drawContours(image=cnt_img, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=3)

# Solution for all questions

img = image.copy()

# count test score
correct = 0

# sorting the contour from top to bottom
question_top_bottom = imutils.contours.sort_contours(question_contours, method='top-to-bottom')[0]

for question, idx in enumerate(range(0, len(question_contours), 5)):

    # extraction of the first five contours and sorting from left to right
    fields = question_top_bottom[idx:idx + 5]
    fields = imutils.contours.sort_contours(fields, method='left-to-right')[0]

    marked = None

    for cnt_idx, contour in enumerate(fields):

        # create a mask with zeros. It's equal to black image
        mask = np.zeros(thresh.shape, dtype='uint8')

        # draw a contour on the mask
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # bitwise_and on the thresh image
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)

        # count the number of zero pixels
        total = cv2.countNonZero(mask)

        if marked is None or total > marked[0]:
            marked = (total, cnt_idx)


    # set default color to red
    color = (0, 0, 255)

    key = ANSWER_KEY[question]

    if key == marked[1]:
        # change the color to green
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(img, [fields[key]], -1, color, 2)

# adding a top border
checked = cv2.copyMakeBorder(
    src=img,
    top=50,
    bottom=0,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT,
    value=(255, 255, 255)
)

score = (correct / 10)

color = (50, 168, 82) if score >= 0.6 else (71, 7, 219)
text = 'Passed' if score >= 0.6 else 'Failed'

cv2.putText(img=checked,
            text=f'{text}: {score * 100}%',
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=color,
            thickness=2)

# write down the image
# cv2.imwrite(filename=checked.png, img=checked)

cv2.imshow('img', checked)
cv2.waitKey(0)
cv2.destroyAllWindows()
