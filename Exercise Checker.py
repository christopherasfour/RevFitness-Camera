from turtle import pd
import cv2
import numpy as np
import time
import PoseDetector as pd

#cap = cv2.VideoCapture("Y2Mate.is - Leaning Cable Lateral Raise-lq7eLC30b9w-720p-1642789733313.mp4")
capture = cv2.VideoCapture(0)
detector = pd.PoseDetector()
count = 0
dir = 0
pTime = 0

bodyPartTargets = {'right arm': [12,14,16], 
					'left arm': [11,13,15], 
					'right back': [12,24,26], 
					'left back': [25,23,11], 
					'right leg': [30,26,24], 
					'left leg': [23,25,29], 
					'right arm back': [14,12,24], 
					'left arm back': [13,11,23]}
                    #MORE TO BE ADDED

exerciseOfChoice1 = 'right arm'
exerciseOfChoice2 = 'left arm'
repCounter = 0

while True:
	success, img = capture.read()
	img = detector.findPose(img, False)
	lmList = detector.findPosition(img, False)
	if len(lmList) != 0:

		
		(p1, p2, p3) = bodyPartTargets[exerciseOfChoice1]
		angle, repCounter = detector.findAngle(img, p3, p2, p1, repCounter)
		# (p1, p2, p3) = bodyPartTargets[exerciseOfChoice2]
		# angle = detector.findAngle(img, p3, p2, p1)


		# # Left Arm
		#angle = detector.findAngle(img, 11, 13, 15,False)
		#per = np.interp(angle, (210, 310), (0, 100))
		#bar = np.interp(angle, (220, 310), (650, 100))
		# print(angle, per)

		
	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime

	cv2.imshow("Image", img)
	cv2.waitKey(1)

