import cv2
import PoseDetector as pd
import time
import numpy as np
#import cgi, cgitb


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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





class Video(object):
    def __init__(self,name):
        self.video = cv2.VideoCapture(0)
        self.detector = pd.PoseDetector()
        self.count = 0
        self.dir = 0
        self.pTime = 0
        self.name = name


        self.bodyPartTargets = {'right arm': [12,14,16], 
                            'left arm': [11,13,15], 
                            'right back': [12,24,26], 
                            'left back': [25,23,11], 
                            'right leg': [30,26,24], 
                            'left leg': [23,25,29], 
                            'right arm back': [14,12,24], 
                            'left arm back': [13,11,23]}


    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret,frame=self.video.read()
        frame = self.detector.findPose(frame, False)
        lmList = self.detector.findPosition(frame, False)
        if len(lmList) != 0:
            if (self.name == "Pushups"):
                exerciseOfChoice1 = 'right arm'
                exerciseOfChoice2 = 'left arm'
            elif (self.name == "Squats"):
                exerciseOfChoice1 = 'right leg'
                exerciseOfChoice2 = 'left leg'
            
            (p11, p21, p31) = self.bodyPartTargets[exerciseOfChoice1]
            angle = self.detector.findAngle(frame, p31, p21, p11)
            (p12, p22, p32) = bodyPartTargets[exerciseOfChoice2]
            angle2 = self.detector.findAngle(frame, p32, p22, p12)
            rep = np.interp(angle, (130, 170), (0, 100))

            #check angle reached
            if (rep == 100):
                if (self.dir == 0):
                    self.count += 0.5
                    self.dir = 1
            if (rep == 0):
                if (self.dir == 1):
                    self.count += 0.5
                    self.dir = 0
            print(self.count)

            cv2.putText(frame, f'{self.count}', (50,100),cv2.FONT_HERSHEY_PLAIN,5, (255,0,0),5)

            # # Left Arm
            #angle = detector.findAngle(img, 11, 13, 15,False)
            #per = np.interp(angle, (210, 310), (0, 100))
            #bar = np.interp(angle, (220, 310), (650, 100))
            # print(angle, per)

                
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime

            #cv2.imshow("Image", frame)
            #cv2.waitKey(1)
        ret,jpg =cv2.imencode('.jpg',frame)
        return jpg.tobytes()