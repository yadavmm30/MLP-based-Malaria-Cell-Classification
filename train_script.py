# installing packages


import os
# os.system("sudo pip install --upgrade pip")
# os.system("sudo pip install opencv-contrib-python")
# os.system("sudo pip install sklearn")
# os.system("sudo pip install --upgrade tensorflow")


# importing packages required
import os
import cv2
import warnings
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.models import load_model
tf.keras.backend.clear_session()
# Ignore warnings
warnings.filterwarnings('ignore')

# Initialization

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-4
N_NEURONS = 300
N_EPOCHS = 40
BATCH_SIZE = 512
RESIZE_TO = 50

# Loading the dataset
directory = os.getcwd() + "/train/"
x_raw, y_raw = [], []
for f in [file for file in os.listdir(directory) if file[-4:] == ".png"]:
    ksize = (5, 5)
    x_org = cv2.imread(directory + f)
    x_raw.append(x_org)
    with open(directory + f[:-4] + ".txt", "r") as imtype:
        l = imtype.read()
        y_raw.append(l)

# dividing the data into train validation and test
x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, random_state=SEED, test_size=0.2, stratify=y_raw)

# train data Augmentation
x_raw, y_raw = [], []
cnt = 0
for i in range(len(x_train)):
    l = y_train[i]
    x_org = x_train[i]
    x_raw.append(x_org)
    y_raw.append(l)
    if l == "schizont" or l == "ring":
        org_flip = cv2.flip(x_org, 0)
        org_mirr = cv2.flip(x_org, 0)
        org_blur = cv2.blur(x_org, ksize)
        org_flip_blur = cv2.blur(org_flip, ksize)
        org_mirr_blur = cv2.blur(org_mirr, ksize)

        rot_90 = cv2.rotate(x_org, cv2.cv2.ROTATE_90_CLOCKWISE)
        rot_90_flip = cv2.flip(rot_90, 0)
        rot_90_mirr = cv2.flip(rot_90, 1)
        rot_90_blur = cv2.blur(rot_90, ksize)
        rot_90_flip_blur = cv2.blur(rot_90_flip, ksize)
        rot_90_mirr_blur = cv2.blur(rot_90_mirr, ksize)

        rot_180 = cv2.rotate(x_org, cv2.ROTATE_180)
        rot_180_flip = cv2.flip(rot_180, 0)
        rot_180_mirr = cv2.flip(rot_180, 1)
        rot_180_blur = cv2.blur(rot_180, ksize)
        rot_180_flip_blur = cv2.blur(rot_180_flip, ksize)
        rot_180_mirr_blur = cv2.blur(rot_180_mirr, ksize)

        rot_270 = cv2.rotate(x_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_270_flip = cv2.flip(rot_270, 0)
        rot_270_mirr = cv2.flip(rot_270, 1)
        rot_270_blur = cv2.blur(rot_270, ksize)
        rot_270_flip_blur = cv2.blur(rot_270_flip, ksize)
        rot_270_mirr_blur = cv2.blur(rot_270_mirr, ksize)

        x_raw.append(org_flip)
        x_raw.append(org_mirr)
        x_raw.append(org_blur)
        x_raw.append(org_flip_blur)
        x_raw.append(org_mirr_blur)
        x_raw.append(rot_90)
        x_raw.append(rot_90_flip)
        x_raw.append(rot_90_mirr)
        x_raw.append(rot_90_blur)
        x_raw.append(rot_90_flip_blur)
        x_raw.append(rot_90_mirr_blur)
        x_raw.append(rot_180)
        x_raw.append(rot_180_flip)
        x_raw.append(rot_180_mirr)
        x_raw.append(rot_180_blur)
        x_raw.append(rot_180_flip_blur)
        x_raw.append(rot_180_mirr_blur)
        x_raw.append(rot_270)
        x_raw.append(rot_270_flip)
        x_raw.append(rot_270_mirr)
        x_raw.append(rot_270_blur)
        x_raw.append(rot_270_flip_blur)
        x_raw.append(rot_270_mirr_blur)

        for i in range(23):
            y_raw.append(l)

    elif l == "trophozoite":
        org_blur = cv2.blur(x_org, ksize)
        rot_90 = cv2.rotate(x_org, cv2.cv2.ROTATE_90_CLOCKWISE)
        rot_90_blur = cv2.blur(rot_90, ksize)
        rot_180 = cv2.rotate(x_org, cv2.ROTATE_180)
        rot_180_blur = cv2.blur(rot_180, ksize)
        rot_270 = cv2.rotate(x_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_270_blur = cv2.blur(rot_270, ksize)

        x_raw.append(org_blur)
        x_raw.append(rot_90)
        x_raw.append(rot_90_blur)
        x_raw.append(rot_180)
        x_raw.append(rot_180_blur)
        x_raw.append(rot_270)
        x_raw.append(rot_270_blur)

        for i in range(7):
            y_raw.append(l)
    elif l == "red blood cell" and cnt<=2000:
        cnt = cnt+1
        org_flip = cv2.flip(x_org, 0)
        org_mirr = cv2.flip(x_org, 0)

        x_raw.append(org_flip)
        y_raw.append(l)
        x_raw.append(org_mirr)
        y_raw.append(l)


def pre_processing(x_raw, y_raw):
    # # Pre-processing the dataset
    le = LabelEncoder()
    le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
    lable = le.transform(y_raw)

    print(lable)

    # To find the mean height and row for resize purpose
    # height = 0
    # width = 0
    # for im in x_raw:
    #     height = height + im.shape[0]
    #     width = width + im.shape[1]
    # height=int(height/len(x_raw))
    # width=int(width/len(x_raw))
    # The output of the above code is 115 for both height and width

    features = []
    for i in x_raw:
        features.append(cv2.resize(i, (RESIZE_TO, RESIZE_TO)))

    # Calculating the number of images of each type
    unique, counts = np.unique(lable, return_counts=True)
    p = dict(zip(unique, counts))
    print(p)

    # converting dataset to numpy
    features, label = np.array(features), np.array(lable)
    return features, label

x_train, y_train = pre_processing(x_raw, y_raw)
x_test, y_test = pre_processing(x_test, y_test)

print("**********success**********")
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

inp_dim = len(x_train[0])
print(inp_dim)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential()
model.add(Dense(N_NEURONS, input_dim=inp_dim, activation="selu"))
model.add(Dense(1024, activation="tanh"))
model.add(Dense(512, activation="selu"))
model.add(Dense(256, activation="tanh"))
model.add(Dense(64, activation="selu"))
model.add(Dense(4, activation="softmax"))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
# saving the model on every epochs
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_madhuriyadav.hdf5", monitor="val_loss", save_best_only=True)])

model = load_model("mlp_madhuriyadav.hdf5")

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
