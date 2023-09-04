import tensorflow as tf
import numpy as np
import os
import json
import librosa
from tqdm import tqdm

DATASET_PATH = '/home/long/Source-Code/SpechCommandAppication/archive/augmented_dataset/augmented_dataset/'
JSON_PATH='data.json'
SAMPLE_RATE = 22050


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure that we're not at the root level
        if dirpath is not dataset_path:
            category = dirpath.split("/")[-1]  # genre/blues => ["genre", "blues"] => blues
            # update mappings
            data["mappings"].append(category)
            pbar = tqdm(filenames)
            # loop through all the filenames and extract MFCCs
            for f in pbar:
                pbar.set_description(f'Processing {category}', refresh=True)
                # Get file path
                file_path = os.path.join(dirpath, f)

                # Load audiofile
                signal, sr = librosa.load(file_path, mono=True)

                # Ensure that all audio files at least 1 sec long
                if len(signal) >= sr:

                    # ensure signal has 1s long
                    signal = signal[:sr]
                    # print(signal.shape)
                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal,
                                                 n_mfcc=n_mfcc, 
                                                 hop_length=hop_length, 
                                                 n_fft=n_fft)
                    
                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    # print('{}: {}'.format(file_path, i-1))

    # Save data to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
