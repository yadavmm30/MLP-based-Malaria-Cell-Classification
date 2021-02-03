# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model
import cv2


def predict(x):
    RESIZE_TO = 50
    x_raw = []
    for f in x:
        x_raw.append(cv2.resize(cv2.imread(f), (RESIZE_TO, RESIZE_TO)))
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    x_raw = np.array(x_raw)
    x_raw = x_raw.reshape(len(x_raw), -1)
    x_raw = x_raw / 255
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model("mlp_madhuriyadav.hdf5")
    y_pred = np.argmax(model.predict(x_raw), axis=1)
    return y_pred, model
