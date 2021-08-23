import geocoder
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
from . import models
from . import form
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView, TemplateView
from django.urls import reverse_lazy, reverse
from django.contrib.auth import login, logout
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.shortcuts import render
from myalert.livedemo import calibration, live, eye_aspect_ratio,run
import pickle
from tensorflow import keras
import requests
import sklearn
# Create your views here.

def loc_ip(requests):
    geolocator = Nominatim(user_agent="Drowsiness Detection")
    url="https://get.geojs.io/v1/ip.json"
    ip_add=requests.get(url).json()['ip']
    print(ip_add)
    url2 = "https://get.geojs.io/v1/ip/geo/" + ip_add + '.json'
    locn = requests.get(url2).json()
    lat=locn['latitude']
    lon=locn['longitude']
    loca=str(lat+","+lon)
    location = geolocator.reverse(loca)
    return location


def SendSMS():
    account_sid = 'ACaf083535b02de893ca8ee1b9c4f4300e'
    auth_token = '71fc0274c660afbb3ad6b34a9e8a53b7'
    client = Client(account_sid, auth_token)
    location=loc_ip(requests)
    message = client.messages.create(
        from_='+15207294645',
        body='Drowsiness Detected! {} needs your help uregntly. His Location:{}'.format("Govind Shukla",location.address),
        to='+918369072464'

    )
    print(message.sid)


class SignUp(CreateView):
    form_class = form.UserCreateForm
    success_url = reverse_lazy("myalert:addprofile")
    template_name = "signup.html"


def get_info(request):
    username=request.user.username
    username=str(username)
    print(username)
    profile = models.Profile.objects.filter(UserName=username)

    sleeps = profile[0].no_of_sleeps
    trips = profile[0].no_of_trips
    name = profile[0].UserName
    if trips == 0:
        accr = 0
    else:
        accr = (trips - sleeps) / trips * 100
    return render(request, 'myprofile.html', {'name': name, 'accr': accr, 'sleeps': sleeps, 'trips': trips})



def StartDrive(request):
    pygame.mixer.pre_init(22050, -16, 2, 64)
    pygame.mixer.init()
    pygame.mixer.quit()
    pygame.mixer.init(22050, -16, 2, 64)

    username = request.user.username
    username = str(username)
    print(username)
    profile = models.Profile.objects.filter(UserName=username)
    no_trips = profile[0].no_of_trips
    no_sleeps = profile[0].no_of_sleeps
    no_trips = no_trips + 1
    models.Profile.objects.filter(UserName=username).update(no_of_trips=no_trips)

    EYE_AR_THRESH = 0.18
    EYE_AR_CONSEC_FRAMES = 3
    sleep_frames = 100


    COUNTER = 0
    TOTAL = 0
    SETS = 0
    CHECK = True

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "static/shape_predictor_68_face_landmarks0.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    print("left eyes start coord")
    print(lStart)

    print("[INFO] starting video stream thread...")
    vs = FileVideoStream("").start()
    fileStream = True
    vs = VideoStream(0).start()
    fileStream = False
    time.sleep(1.0)

    while True:

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

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

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
                        models.Profile.objects.filter(UserName=username).update(no_of_sleeps=no_sleeps)
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
        key = cv2.waitKey(300) & 0xFF

        # if the ESC key was pressed, break from the loop
        if key == 27:
            break

    cv2.destroyAllWindows()
    vs.stop()
    return render(request, 'index.html')


def Drive(request):
    if request.method=="POST":
        model=request.POST["mlmodel"]
    if model =="Greedy":
        StartDrive(request)
    elif model=="Logistics":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/drowsy_logistic'
        with open(model_path, 'rb') as f:
            clf_logistic = pickle.load(f)
        run(request,clf_logistic)
    elif model=="KNN":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_KNN'
        with open(model_path, 'rb') as f:
            clf_KNN = pickle.load(f)
        run(request, clf_KNN)
    elif model=="MLP":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_mlp1'
        with open(model_path, 'rb') as f:
            clf_MLP = pickle.load(f)
        run(request, clf_MLP)
    elif model=="NaiveBayes":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_NaiveBayes'
        with open(model_path, 'rb') as f:
            clf_NB = pickle.load(f)
        run(request, clf_NB)
    elif model=="RandomForest":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_RandomForest'
        with open(model_path, 'rb') as f:
            clf_RF = pickle.load(f)
        run(request, clf_RF)
    elif model=="XGB":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_XGB'
        with open(model_path, 'rb') as f:
            clf_XGB = pickle.load(f)
        run(request, clf_XGB)
    elif model=="CNN":
        model_path = '/Users/govindshukla/PycharmProjects/DriverAlert/MyDriverAlert/static/Data/my_CNNn'
        cnn = keras.models.load_model(model_path)
        run(request, cnn)




class add_profile(CreateView):
    form_class = form.AddProfileForm
    success_url = reverse_lazy("myalert:login")
    template_name = "addprofile.html"
