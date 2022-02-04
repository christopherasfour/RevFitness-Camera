import cv2
import mediapipe
import time
import math

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 minDetectionCon=False, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.minDetectionCon = minDetectionCon
        self.trackCon = trackCon

        self.mpDraw = mediapipe.solutions.drawing_utils
        self.mpPose = mediapipe.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.minDetectionCon, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if (self.results.pose_landmarks):
            if (draw):
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, repCounter, draw=True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if (angle < 0):
            angle += 360
        
        

        # Draw
        if (draw):
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            if (angle >= 150 and angle <= 160):
                cv2.circle(img, (x1, y1), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0,255,0), 2)
                cv2.circle(img, (x2, y2), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0,255,0), 2)
                cv2.circle(img, (x3, y3), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0,255,0), 2)
                repCounter = repCounter + 1
                print(repCounter)
            if (angle >= 190 and angle <= 195):
                cv2.circle(img, (x1, y1), 10, (255,0,0), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (255,0,0), 2)
                cv2.circle(img, (x2, y2), 10, (255,0,0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255,0,0), 2)
                cv2.circle(img, (x3, y3), 10, (255,0,0), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (255,0,0), 2)
                
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 255), 2)
        return (angle, repCounter)

def main():
    pTime = 0
    # pDetector = PoseDetector()
    # while True:
    #     success, img = cap.read()
    #     img = pDetector.findPose(img)
    #     draw = False
    #     lmList = pDetector.findPosition(img, draw)
    #     if len(lmList) != 0:
    #         print(lmList[14])
    #         cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

    #     cTime = time.time()
    #     fps = 1 / (cTime - pTime)
    #     pTime = cTime

    #     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    #                 (255, 0, 0), 3)

    #     cv2.imshow("Image", img)
    #     cv2.waitKey(1)


if __name__ == "__main__":
    main()