import predict
import pandas as pd
import argparse
import dlib
import uuid
import os

if __name__ == "__main__":

    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        'dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor(
        'dlib_models/shape_predictor_5_face_landmarks.dat')

    f_id = str(uuid.uuid4())
    SAVE_DETECTED_AT = "detected_faces/" + f_id
    predict.ensure_dir(SAVE_DETECTED_AT)  # 디렉토리 생성
    nas = "detected_faces/race_Asian_face0.jpg"
    imgs = [nas]
    predict.detect_face(imgs, SAVE_DETECTED_AT, cnn_face_detector, sp)
    #print("detected faces are saved at ", SAVE_DETECTED_AT)
    model_7= predict.make_model7()
    model_4= predict.make_model4()
    predict.predidct_age_gender_race("test_outputs.csv", SAVE_DETECTED_AT,model_7,model_4)
    #print('end')

    f = open("test_outputs.csv", "r")
    read = f.read()
    split = read.split(",")
    race = split[9]
    race4 = split[10]
    gender = split[11]
    age = split[12]

    arr = [race, race4, gender, age]
    print(arr)
