from flask import Flask, Response, render_template, url_for,request, redirect
from camera import Video
import os
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
import argparse
import asyncio
from cgitb import html
import json
import logging
import os
from pathlib import Path
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

app = Flask(__name__)

workoutofChoice = ''
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
logger = logging.getLogger("pc")
pcs = set()
   
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/workout/')
def workout():
    return render_template('Workouts.html')

@app.route('/workoutCamera/',methods=['GET', 'POST'])
def workoutCamera():
    if request.method == 'POST':
        form = request.form
        if request.form['button'] == 'Lateral Raises':
            workoutofChoice = 'Lateral Raises'
        elif request.form['button'] == 'Bicep Curls':
            workoutofChoice = 'Bicep Curls'
        elif request.form['button'] == 'Squats':
            workoutofChoice = 'Squats'

    return render_template('Workout-Camera.html', name=workoutofChoice)

@app.route('/home/')
def home():
    return render_template('Home.html')

@app.route('/about/')
def about():
    return render_template('About-Us.html')

@app.route('/login/')
def login():
    return render_template('Login-Page.html')

@app.route('/signup/')
def signup():
    return render_template('Signup-Page.html')

def gen(camera, workoutofChoice):
    counter = 0 
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.85) as pose:
        while camera.isOpened():
            ret, frame = camera.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Get coordinates
                leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                rightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Curl counter logic
                if (workoutofChoice == 'Lateral Raises'):
                    rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    angle_rightarm = calculateAngle(rightHip, rightShoulder, rightElbow)
                    angle_leftarm = calculateAngle(leftHip, shoulder, elbow)
                    hit1 = False
                    if angle_rightarm > 99 and angle_leftarm > 99:
                        stage = "Down"
                    if (angle_rightarm < 35 and angle_leftarm < 35) and stage =='Down':
                        stage="Up"
                        counter +=1
                        hit1 = True
                        successColor = (0,255,0)
                        print(counter)
                         
                    
                    cv2.putText(image, str(angle_rightarm), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                            # Render curl counter
                    # Setup status box
                    if (hit1):

                        cv2.rectangle(image, (0,0), (225,73), successColor, -1)
                    else:
                        cv2.rectangle(image, (0,0), (225,73), (255,0,0), -1)
                    
                    # Rep data
                    # cv2.putText(image, 'Reps', (15,12), 
                    #             cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    # cv2.putText(image, 'STAGE', (19,60), 
                    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (90,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3), 
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3) 
                                            )    
                elif (workoutofChoice == 'Squats'):
                    hit1 = False
                    angle_knee = calculateAngle(leftHip, knee, ankle) #Knee joint angle
                    angle_hip = calculateAngle(shoulder, rightHip, knee)
                    hip_angle = 180-angle_hip
                    knee_angle = 180-angle_knee
                    print(angle_knee)
                    
                    if angle_knee > 159:
                        stage = "Down"
                    if angle_knee <= 90 and stage =='up':
                        stage="Up"
                        hit1 = True
                        successColor = (0,255,0)
                        counter +=1

                    # Visualize angle
                    cv2.putText(image, str(angle_knee), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    if (hit1):

                        cv2.rectangle(image, (0,0), (225,73), successColor, -1)
                    else:
                        cv2.rectangle(image, (0,0), (225,73), (255,0,0), -1)
                                # Render curl counter
                    # Setup status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Rep data
                    # cv2.putText(image, 'Reps', (15,12), 
                    #             cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    # cv2.putText(image, 'STAGE', (19,60), 
                    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (90,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3), 
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3) 
                                            )    
                elif (workoutofChoice == 'Bicep Curls'):
                    hit1 = False
                    successColor = (0,255,0)
                    angle = calculateAngle(shoulder, elbow, wrist) 
                    
                    if angle > 160:
                        stage = "Down"
                    if angle < 30 and stage =='Down':
                        stage="Up"
                        hit1 = True
                        counter +=1
                    

                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                            # Render curl counter
                    # Setup status box
                    if (hit1):

                        cv2.rectangle(image, (0,0), (225,73), successColor, -1)
                    else:
                        cv2.rectangle(image, (0,0), (225,73), (255,0,0), -1)

                    # Rep data
                    # cv2.putText(image, 'Reps', (15,12), 
                    #             cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    # cv2.putText(image, 'STAGE', (19,60), 
                    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (90,60), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2, cv2.LINE_AA)
                    
                    
                    
                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3), 
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3) 
                                            )    
            
            except Exception as e:
                print(e)
                cv2.rectangle(image, (0,0), (225,73), (0,0,255), -1)
                cv2.putText(image, 'BODY NOT FOUND', (15,15), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 2, cv2.LINE_AA)
            
            stat,frame =cv2.imencode('.jpg',image)
            yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')           

@app.route('/video/<variable>')
def video(variable):
    print("name =" + variable)
    #return Response(gen(Video(variable)),  //right solution
    return Response(gen(initializeCamera(), variable),
    mimetype='multipart/x-mixed-replace; boundary=fr')


# @app.route('/offer', methods=['GET','POST'])
# async def offer():
#     params = request.get_json()
#     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

#     pc = RTCPeerConnection()
#     pc_id = "PeerConnection(%s)" % uuid.uuid4()
#     pcs.add(pc)

#     def log_info(msg, *args):
#         logger.info(pc_id + " " + msg, *args)

#     #log_info("Created for %s", request.remote)

#     # prepare local media
#     #player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
#     recorder = MediaBlackhole()

#     @pc.on("datachannel")
#     def on_datachannel(channel):
#         @channel.on("message")
#         def on_message(message):
#             if isinstance(message, str) and message.startswith("ping"):
#                 channel.send("pong" + message[4:])

#     @pc.on("iceconnectionstatechange")
#     async def on_iceconnectionstatechange():
#         log_info("ICE connection state is %s", pc.iceConnectionState)
#         if pc.iceConnectionState == "failed":
#             await pc.close()
#             pcs.discard(pc)

#     @pc.on("track")
#     def on_track(track):
#         log_info("Track %s received", track.kind)
        
#         if track.kind == "video":
#             local_video = VideoTransformTrack(
#                 track, transform=params["video_transform"]
#             )
#             pc.addTrack(local_video)

#         @track.on("ended")
#         async def on_ended():
#             log_info("Track %s ended", track.kind)
#             await recorder.stop()

#     # handle offer
#     await pc.setRemoteDescription(offer)
#     await recorder.start()

#     # send answer
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     x = json.dumps(
#             {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
#     print(x)
#     return Response(
#         x, content_type="application/json")

#     return Response()

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''
    a = np.array(landmark1) # First
    b = np.array(landmark2) # Mid
    c = np.array(landmark3) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

def initializeCamera():
    camera_video = cv2.VideoCapture(0)
    return camera_video

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="WebRTC audio / video / data-channels demo"
    # )
    # parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    # parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    # parser.add_argument(
    #     "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    # )
    # parser.add_argument(
    #     "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    # )
    # parser.add_argument("--verbose", "-v", action="count")
    # parser.add_argument("--write-audio", help="Write received audio to a file")
    # args = parser.parse_args()

    # if args.verbose:
    #     logging.basicConfig(level=logging.DEBUG)
    # else:
    #     logging.basicConfig(level=logging.INFO)

    # if args.cert_file:
    #     ssl_context = ssl.SSLContext()
    #     ssl_context.load_cert_chain(args.cert_file, args.key_file)
    # else:
    #     ssl_context = None
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
