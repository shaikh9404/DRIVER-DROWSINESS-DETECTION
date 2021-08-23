import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance
import math
import pandas as pd
import numpy as np
import pickle
import warnings
from django.shortcuts import render


p = "static/shape_predictor_68_face_landmarks0.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def average(y_pred):
    for i in range(len(y_pred)):
        if i % 240 == 0 or (i+1) % 240 == 0:
            pass
        else:
            average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
            if average >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred

def model(landmark,clf):
    features = pd.DataFrame(columns=["EAR", "MAR", "Circularity", "MOE"])
    eye = landmark[36:68]
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    cir = circularity(eye)
    mouth_eye = mouth_over_eye(eye)
    df = features.append({"EAR": ear, "MAR": mar, "Circularity": cir, "MOE": mouth_eye}, ignore_index=True)
    df["EAR_N"] = (df["EAR"] - mean["EAR"]) / std["EAR"]
    df["MAR_N"] = (df["MAR"] - mean["MAR"]) / std["MAR"]
    df["Circularity_N"] = (df["Circularity"] - mean["Circularity"]) / std["Circularity"]
    df["MOE_N"] = (df["MOE"] - mean["MOE"]) / std["MOE"]
    Result = clf.predict(df)
    if Result == 1:
        Result_String = "Drowsy"
    else:
        Result_String = "Alert"

    return Result_String, df.values

# Callibration

def calibration():
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 400)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    data = []
    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 0)
        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            data.append(shape)
            cv2.putText(image, "Calibrating...", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("CalibrationOutput", image)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    features_test = []
    for d in data:
        eye = d[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        features_test.append([ear, mar, cir, mouth_eye])

    features_test = np.array(features_test)
    x = features_test
    y = pd.DataFrame(x, columns=["EAR", "MAR", "Circularity", "MOE"])
    df_means = y.mean(axis=0)
    df_std = y.std(axis=0)
    return df_means, df_std

# live demo
def live(clf):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 400)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cap = cv2.VideoCapture(0)
    data = []
    result = []
    while True:
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            Result_String, features = model(shape,clf)
            cv2.putText(image, Result_String, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            data.append(features)
            result.append(Result_String)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("Output", image)
        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    return data, result


def run(request,model):
    mean, std = calibration()
    data, result = live(model)
    print(result)
    return render(request, 'index.html')



