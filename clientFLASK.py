import requests
import os
URL = 'http://127.0.0.1/predict'
number=2
if __name__ == "__main__":
    # list all files in test folder
    list_files = os.listdir('test')
    for file in list_files:
        audio = open('test/' + file, 'rb')
        values = {'file': ('test/' + file, audio, 'audio/wav')}
        # Send request to URL with values is audio file
        response = requests.post(URL, files=values)
        data = response.json()
        print(f'Ground truth: {file} Predicted keyword: {data["keyword"]}')
    
