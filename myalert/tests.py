from geopy.geocoders import Nominatim
import pygame
import cv2
import dlib
import time
import imutils
import argparse
import numpy as np
from twilio.rest import Client
from imutils import face_utils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist

from myalert.livedemo import calibration, live, eye_aspect_ratio,run
import pickle
from tensorflow import keras
import requests

def StartDrive(request):
    pygame.mixer.pre_init(22050, -16, 2, 64)
    pygame.mixer.init()
    pygame.mixer.quit()
    pygame.mixer.init(22050, -16, 2, 64)

    # profile = models.Profile.objects.filter(id=1)
    no_trips = 0
    no_sleeps = 0
    no_trips = no_trips + 1
    # models.Profile.objects.filter(id=1).update(no_of_trips=no_trips)

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    sleep_frames = 100

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    SETS = 0
    CHECK = True
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "static/shape_predictor_68_face_landmarks0.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    print("left eyes start coord")
    print(lStart)

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream("").start()
    fileStream = True
    vs = VideoStream(0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
    time.sleep(1.0)

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= sleep_frames:

                    if CHECK:
                        no_sleeps = no_sleeps + 1
                        # models.Profile.objects.filter(
                        #     id=1).update(no_of_sleeps=no_sleeps)
                        CHECK = False
                    COUNTER = 0
                    SETS += 1
                    print("DONT SLEEP")
                    pygame.mixer.music.load("static/file.mp3")
                    pygame.mixer.music.play()
                    if SETS >= 4:
                        SendSMS()
                        SETS = 0
                        print("msg sent")

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:

                pygame.mixer.music.stop()

                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                    # reset the eye frame counter
                    COUNTER = 0
                    SETS = 0

            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(
                ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Frames: {}".format(COUNTER), (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    # return render(request, 'index.html')

StartDrive(requests)