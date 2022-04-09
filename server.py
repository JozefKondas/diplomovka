from __future__ import print_function
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
from matplotlib import pyplot as plt
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

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

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
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

    # Create strategy
   strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=1,
        min_available_clients=10,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
   fl.server.start_server("[::]:8080", config={"num_rounds": 4}, strategy=strategy)

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

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
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
    x_val, y_val = x_train[0:50], y_train[0:50]


    # The `evaluate` function will be called after every round
    
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()