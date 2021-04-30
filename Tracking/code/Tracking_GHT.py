import matplotlib.pyplot as plt
import numpy as np
from numpy import shape
import cv2
from collections import defaultdict

roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True

def gradient_orientation(image):

    # CALCULATE THE GRADIENT ORIENTATION FOR EDGE POINTS IN THE IMAGE

    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    gradient = np.arctan2(dy, dx) * 180 / np.pi

    return gradient

def pre_processing(roi):

    # EXETRACT THE EDGES OF ROI

    roi_hough = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(roi_hough, cv2.CV_32F, 1, 0)  # gradient of x
    grad_y = cv2.Sobel(roi_hough, cv2.CV_32F, 0, 1)  # gradient of y
    roi_hough = cv2.convertScaleAbs(np.hypot(grad_x, grad_y))
    roi_hough[np.where(roi_hough[:, :] < threshold)] = 0  # set these pixels valeurs < threshold => 0

    return roi_hough

def create_RTable(roi_hough):

    # CREATES THE R-TABLE

    # Computes the center of the roi_hough as reference point
    origin = [int(shape(roi_hough)[0] / 2), int(shape(roi_hough)[1] / 2)]

    gradient = gradient_orientation(roi_hough)

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(roi_hough):
        if value:
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table

def match_hough(edges, r_table):

    # MATCH WITH THE R-TABLE

    gradient = gradient_orientation(edges)
    accumulator = np.zeros(edges.shape)
    for (i, j), value in np.ndenumerate(edges):
        if value:
            for r in r_table[gradient[i, j]]:
                accum_i, accum_j = i + r[0], j + r[1]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[int(accum_i), int(accum_j)] += 1

    return accumulator

### PATH OF VIDEO ###
cap = cv2.VideoCapture('./Test-Videos/Antoine_Mug.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Woman.mp4')

### DEFINE ROI ###
ret, frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

while True:
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    if (roi_defined):
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    else:
        frame = clone.copy()
    if key == ord("q"):
        break
track_window = (r, c, h, w)
roi = clone[c:c + w, r:r + h]

# treshold for hiding the pixels that orientation isn't significant
threshold = 30

#  MODÈLE IMPLICITE
roi_hough = pre_processing(roi)
cv2.imshow('Model', roi_hough)
r_table = create_RTable(roi_hough)

cpt = 0
while (1):
    ret, frame = cap.read()
    if ret == True:

        # ORIENTATION LOCALE
        img = pre_processing(frame)

        # MATCH R-TABLE AND img
        accumulator = match_hough(img, r_table)

        # FIND THE MAXIMUM VALUE OF HOUGH TRANSFORM
        x_acc, y_acc = np.unravel_index(np.argmax(accumulator), shape(accumulator))
        frame_tracked = cv2.rectangle(frame, (y_acc - int(h / 2), x_acc - int(w / 2)),
                                      (y_acc + int(h / 2), x_acc + int(w / 2)), (255, 0, 0), 2)

        cv2.imshow('Sequence', frame_tracked)

        # #Plots the Hough transoform
        # plt.figure("Hough")
        # plt.imshow(accumulator)
        # plt.draw()
        # plt.pause(0.0001)

        cpt += 1
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_roi_hough%04d.png' % cpt, roi_hough)
            cv2.imwrite('Frame_%04d.png' % cpt, frame_tracked)
            cv2.imwrite('Frame_img%04d.png' % cpt, img)
            cv2.imwrite('Frame_accu%04d.png' % cpt, accumulator)
    else:
        break