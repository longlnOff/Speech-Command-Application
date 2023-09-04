import requests
import os
URL = 'http://127.0.0.1/predict'
number=2
TEST_AUDIO_FILE_PATH = 'test/cat'+str(number)+'.wav'
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    audio = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {'file': (TEST_AUDIO_FILE_PATH, audio, 'audio/wav')}
    # Send request to URL with values is audio file
    response = requests.post(URL, files=values)
    data = response.json()
    print(f'Predicted keyword: {data["keyword"]}')
    
