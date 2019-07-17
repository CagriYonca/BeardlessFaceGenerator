# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])

faces = []
faces.append(np.expand_dims(image, 0))
faces = np.asarray(faces)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	center_eyes = shape[27].astype(np.int)
	eyes_d = np.linalg.norm(shape[36] - shape[45])
	face_size_x = int(eyes_d * 2)

	d = (shape[45] - shape[36]) / eyes_d
	a = np.rad2deg(np.arctan2(d[1], d[0]))
	scale_factor = float(256) / float(face_size_x * 1.5)

	M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1] + 15), a, scale_factor),[[0,0,1]], axis=0)
	M1 = np.array([[1, 0, -center_eyes[0] + 128.], 
		[0, 1, -center_eyes[1] + 128.], [0, 0, 1]])
	M = M1.dot(M)[:2]

	try:
		face = cv2.warpAffine(image, M, (256, 256))
		faces.append(face)
	except:
		continue

cv2.imwrite("new" + args["image"], face)
cv2.imshow("Output", face)
cv2.waitKey(0)
