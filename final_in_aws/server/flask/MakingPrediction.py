import tensorflow as tf
import os
import json
import librosa
import numpy as np
import onnxruntime
import onnx
DATA_PATH = 'data.json'
MODEL_PATH = 'model.h5'
MODEL_ONNX_PATH = 'model.onnx'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import os
# print(os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'))
# print(os.environ.get('CUDA_HOME'))
# open
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

class _Keyword_Spotting_Service:
    model = None
    model_onnx = None
    _mapping = data['mappings'].copy()
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2d MFCCs array into 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword
    
    def predict_onnx(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2d MFCCs array into 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        input_name = self.model_onnx.get_inputs()[0].name
        output_name = self.model_onnx.get_outputs()[0].name
        predictions = self.model_onnx.run([output_name], {input_name: MFCCs.astype(np.float32)})
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        
        return predicted_keyword

    
    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)
        # ensure consistency in the audio file length
        if len(signal) >= 22050:
            signal = signal[:22050]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, 
                                         n_fft=n_fft, hop_length=hop_length)
            
            return MFCCs.T
    
def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance  = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model      = tf.keras.models.load_model(MODEL_PATH)
        _Keyword_Spotting_Service.model_onnx = onnxruntime.InferenceSession(MODEL_ONNX_PATH)
    return _Keyword_Spotting_Service._instance



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    kss = Keyword_Spotting_Service()
    labels = kss.predict('/home/long/Source-Code/SpechCommandAppication/archive/augmented_dataset/augmented_dataset/tree/2.wav')
    print(labels)