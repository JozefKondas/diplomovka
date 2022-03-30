import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np # linear algebra
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats

import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

##############this is function for my dataset, you probably won't need it #####################

def get_frames(df, frame_size, hop_size):
    N_FEATURES = 3
    frames = []
    labels = []
    for i in range(0,len(df )- frame_size, hop_size):
        x = df['x'].values[i: i+frame_size]
        y = df['y'].values[i: i+frame_size]
        z = df['z'].values[i: i+frame_size]
        label = stats.mode(df['label'][i: i+frame_size])[0][0]
        frames.append([x,y,z])
        labels.append(label)
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    return frames, labels

#####################################

if __name__ == "__main__":
    # Load and compile Keras model
    model = Sequential()
    model.add(Conv2D(256, (2,2), activation = 'relu', input_shape = (80, 3, 1)))
    model.add(Dropout(0.1))
    model.add(Conv2D(512, (2,2), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Dense(1024, activation = 'relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(18, activation='softmax'))
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset

#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    dataframe = pd.read_csv("f1.csv", header=None)
    dataset = dataframe.values

    final = pd.DataFrame(data=dataset, columns=['x','y','z','label'])

    final = final.iloc[1: , :]
    #X = dataset[:, 0:3]   #inputs
    #X = dataset[:,0:4]
    #Y = dataset[:,3]      #class (outputs)
    #Y = int(Y) 

    x = final[['x','y','z']]
    y = final['label'].astype('float')
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    scaled_x = pd.DataFrame(data=x, columns=['x','y','z'])
    scaled_x['label'] = y.values

    print(scaled_x)

    Fs=20
    frame_size = Fs*4 #80
    hop_size = Fs*2 #40

    x,y = get_frames(scaled_x, frame_size, hop_size)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state = 0)
    x_train = x_train.reshape(x_train[:, :, :, np.newaxis].shape)
    x_test = x_test.reshape(x_test[:, :, :, np.newaxis].shape)
    # Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client , need to add IP adress:PORT of your server##################
    fl.client.start_numpy_client("20.120.29.169:5040", client=CifarClient())
