import tensorflow 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

#print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
#print(device_lib.list_local_devices())


X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


model = Sequential()
#Adding layers
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
#Compiling the model for training  
opt = Adamax(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

print(model.summary())


history = model.fit(X_train, y_train,
		validation_data=(X_val,y_val),
		epochs=125,
		batch_size=128)

model.save("bach_epoch125_bs128_allNotes_lr01.h5")

def visualize_history(history):
    history = history.history
    plt.figure(0)
    plt.plot(history['accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    #plt.savefig('chopin_epoch200_bs64_train_accuracy_del500_notes')

    np.save("bach_epoch125_bs128_allNotes_lr01_train_accuracy.npy", history["accuracy"])
    np.save("bach_epoch125_bs128_allNotes_lr01_val_accuracy.npy", history["val_accuracy"])

    plt.figure(1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #plt.savefig('chopin_epoch200_bs64_train_loss_del500_notes')
    np.save("bach_epoch125_bs128_allNotes_lr01_train_loss.npy", history["loss"])
    np.save("bach_epoch125_bs128_allNotes_lr01_val_loss.npy", history["val_loss"])

visualize_history(history)




