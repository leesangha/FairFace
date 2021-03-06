import os
import io
import uuid
import shutil
import sys

import threading
import time
from queue import Empty, Queue

from flask import Flask, render_template, flash, send_file, request, jsonify, url_for
import numpy as np
import dlib
from predict import detect_face, predidct_age_gender_race, make_model7, make_model4
#################################################################
app = Flask(__name__, template_folder="templates", static_url_path="/static")

DATA_FOLDER = "detected_faces"
# Init Cartoonizer and load its weights
model_7 = make_model7()
model_4 = make_model4()

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
##################################################################
# pre-train
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    'dlib_models/mmod_human_face_detector.dat')
sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
# run

def run(input_file, file_type, f_path):
    try:
        f_name = str(uuid.uuid4())
        save_path = f_path + '/' + f_name + '.jpg'
        file_name = f_name+'.jpg'

        # Original Image Save
        input_file.save(save_path)
        # Run model
        imgs = [save_path]
        detect_face(imgs, f_path, cnn_face_detector, sp)
        #print('detect_face end')
        
        os.remove(save_path)  # 삭제
        if os.path.isfile(save_path):
           print('notremoved : ' + save_path)

        arr = predidct_age_gender_race(
           "test_outputs.csv", f_path, model_7, model_4)
        return arr

    except Exception as e:
        print(e)
        return 500
# Queueing


def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []
            while not (
                len(requests_batch) >= BATCH_SIZE  # or
            ):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue

                for request in requests_batch:
                    request['output']=run(request['input'][0],request['input'][1],request['input'][2])

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)


# Thread Start
threading.Thread(target=handle_requests_by_batch).start()


@app.route("/")
def main():
    
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        if requests_queue.qsize() != 0 :
            print('too many requests')
            return jsonify({"message": "Too Many Requests"}), 429
        #print('current qsize')
        #print(requests_queue.qsize())
        input_file = request.files["source"]
        file_type = request.form["file_type"]
        if file_type == "image":
            if input_file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
                return jsonify({"message": "Only support jpeg, jpg or png"}), 400

        # mkdir and path setting
        f_id = str(uuid.uuid4())
        f_path = os.path.join(DATA_FOLDER, f_id)
        os.makedirs(f_path, exist_ok=True)

        req = {"input": [input_file, file_type, f_path]}
        requests_queue.put(req)
        #print('push queue')
        # Thread output response
        while "output" not in req:
            time.sleep(CHECK_INTERVAL)

        if req["output"] == 500:
            return jsonify({"error": "Error! Please upload another file"}), 500

        result = req["output"]
        #output check 
        print(result)
        shutil.rmtree(f_path)
        array = result.split(",")
        return jsonify(race7=array[0], race4=array[1], gender=array[2], age=array[3]), 200

    except Exception as e:
        print(e)
        return jsonify({"message": "Error! Please upload another file"}), 400


@app.route("/health",methods=["GET"])
def health():
    return "ok",200


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=80)
