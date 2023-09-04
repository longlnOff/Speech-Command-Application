"""
server

client -> POST request -> server -> predict -> response -> client
"""

from MakingPrediction import Keyword_Spotting_Service
import tensorflow as tf
import numpy as np
import random
import os
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)

kss = Keyword_Spotting_Service()
# labels = kss.predict('/home/long/Source-Code/SpechCommandAppication/archive/augmented_dataset/augmented_dataset/tree/2.wav')
labels = kss.predict('2.wav')

print(labels)


@app.route('/predict', methods=['POST'])
def predict():
    # get audio file and save it
    audio_file = request.files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)
    # invoke kWS sevice
    ksw_service = Keyword_Spotting_Service()
    # predict using model
    predicted_keyword = ksw_service.predict_onnx(file_name)

    # # remove temporary audio file
    os.remove(file_name)

    # # send the predicted keyword in json format
    data = {'keyword': predicted_keyword}

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False)