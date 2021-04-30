import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


# cap = cv2.VideoCapture('./Test-Videos/Antoine_Mug.mp4')
cap = cv2.VideoCapture('./Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('./Test-Videos/VOT-Woman.mp4')R

opt= 1

while True:
	ret, frame = cap.read()
	if ret == True:
		if opt == 1:
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			g_x = cv2.Sobel(img, cv2.CV_32F, 1, 0) # gradient of x orientation
			g_y = cv2.Sobel(img, cv2.CV_32F, 0, 1) # gradient of y orientation

			# Norme of gradient
			# mapping the norme of gradient between 0 and 255 in order to show the figure
			N = cv2.convertScaleAbs(np.hypot(g_x, g_y))
			img2 = cv2.cvtColor(N, cv2.COLOR_GRAY2BGR)
			# set threshold and set the value low threshold to [0, 0, 255]
			img2[np.where(((img2[:, :, :] < 75)).all(axis=2))] = [0, 0, 255]

			plt.subplot(231)
			plt.imshow(frame[:,:,::-1], vmin=0.0, vmax=255.0)
			plt.title("Original")

			plt.subplot(232)
			plt.imshow(g_x, cmap = 'gray',vmin = -255.0,vmax = 255.0)
			plt.title("Gradient en x")

			plt.subplot(233)
			plt.imshow(g_y, cmap='gray', vmin=-255.0, vmax=255.0)
			plt.title("Gradient en y")

			plt.subplot(234)
			plt.imshow(frame[:,:,::-1], vmin=0.0, vmax=255.0)
			plt.title("Original")

			plt.subplot(235)
			plt.imshow(N, cmap='gray', vmin=0.0, vmax=255.0)
			plt.title("Norme du gradient")

			plt.subplot(236)
			plt.imshow(img2[:,:,::-1], cmap='gray', vmin=0.0, vmax=255.0)
			plt.title("Orientation")
			plt.show()
			# plt.draw()
			# plt.pause(0.0001)

			k = cv2.waitKey(0) & 0xff
			if k == 27:
				break
		opt += 1

	else:
		break

cv2.destroyAllWindows()
cap.release()
