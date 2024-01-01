import tensorflow 
import numpy as np 
import pandas as pd 
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)

model = load_model("chopin_epoch200_bs256_all_notes.h5")

'''
train_acc = np.load("chopin_epoch125_bs128_allNotes_lr01_train_accuracy.npy")
val_acc = np.load("chopin_epoch125_bs128_allNotes_lr01_val_accuracy.npy")
train_loss = np.load("chopin_epoch125_bs128_allNotes_lr01_train_loss.npy")
val_loss = np.load("chopin_epoch125_bs128_allNotes_lr01_val_loss.npy")

chopin=np.load("chopin_epoch125_bs128_allNotes_lr01_val_accuracy.npy")
beet=np.load("beet_epoch125_bs128_allNotes_lr01_val_accuracy.npy")
bach=np.load("bach_epoch125_bs128_allNotes_lr01_val_accuracy.npy")


plt.figure(0)
plt.plot(chopin, label='Chopin')
plt.plot(beet, label='Beethoven')
plt.plot(bach, label='Bach')
plt.title('Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')

plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
'''
#plt.legend()
#plt.show()

#plt.savefig('chopin_epoch10_bs32_allNotes_lr1_train_accuracy')



X = np.load("X.npy")
y = np.load("y.npy")
all_notes = np.load("all_notes.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
all_notes_test = np.load("all_notes_test.npy")
symb = sorted(list(set(all_notes)))
L_symb = len(symb)
reverse_mapping = dict((i, c) for i, c in enumerate(symb))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_test)


def chords_n_notes(Snippet):
    Melody = []
    offset = 0 
    for i in Snippet:
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)            
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        offset += 1
    Melody_midi = stream.Stream(Melody)   
    return Melody_midi


def Malody_Generator(Note_Count):
    seed = X_val[np.random.randint(0,len(X_val)-1)]

    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = seed.reshape(1,40,1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(L_symb)   
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]

    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)   
    return Music,Melody_midi


Music_notes, Melody = Malody_Generator(40)
Melody.write("midi", "chopin_generated_1.mid")
#real = [reverse_mapping[char] for char in X_test[40:81]]
print(all_notes_test[40:81])
print(Music_notes)

