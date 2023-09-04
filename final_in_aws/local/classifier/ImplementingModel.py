import tensorflow as tf
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
DATA_PATH           = 'data.json'
SAVED_MODEL_PATH    = 'model.h5'
LEARNING_RATE       = 0.0001
BATCH_SIZE          = 48
NUMBER_OF_EPOCHS    = 40
NUMBER_OF_KEYWORDS  = 30
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def load_dataset(data_path):
    # load from json
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y


def get_data_split(data_path):
    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    # convert inputs from 2D-arrays to 3D-arrays
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test



def build_model(input_shape, lr, number_of_classes, error='sparse_categorical_crossentropy'):
    # Build network
    model = tf.keras.Sequential()

    # conv layer 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # layer norm
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # conv layer 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # layer norm
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # conv layer 3
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # layer norm
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output
    model.add(tf.keras.layers.Flatten())

    # dense layer 1
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # dropout 
    model.add(tf.keras.layers.Dropout(0.3))
    
    # dense layer 2 (softmax)
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss=error,
                  metrics=['accuracy'])
    
    # show model summary
    model.summary()

    return model


   

def train():

    # load train/validation/test data splits
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_split(DATA_PATH)

    # Build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # coefficients=13, # channels=1)
    model = build_model(input_shape=input_shape, 
                        lr=LEARNING_RATE,
                        number_of_classes=NUMBER_OF_KEYWORDS)

    # Train the model
    model.fit(X_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X_validation, y_validation))
    
    # Evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # Save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    train()