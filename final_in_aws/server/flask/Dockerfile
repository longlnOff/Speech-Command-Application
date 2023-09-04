# Pull; tensorflow image with python3
FROM tensorflow/tensorflow:2.12.0

# Set the working directory to /app
WORKDIR /app

# Transfer the current directory contents into the container at /app
ADD . /app

# # install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg sox ffmpeg libcairo2 libcairo2-dev


RUN python3 -m pip install --upgrade pip

# Isntall python packages
RUN pip install -r requirements.txt

# Start uWSGI using our uWSGI ini file
CMD ["uwsgi", "app.ini"]