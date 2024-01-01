import numpy as np 
import pandas as pd
import music21
from music21 import *
import sys
import os
import tensorflow 
from collections import Counter

filepath = "../data mining/music-midi/chopin/"
all_midis = []
for i in os.listdir(filepath):
		tr = filepath+i
		midi = converter.parse(tr)
		all_midis.append(midi)

def extract_notes(file):
	notes = []
	pick = None
	for j in file:
		songs = instrument.partitionByInstrument(j)
		for part in songs.parts:
			pick = part.recurse()
			for element in pick:
				if isinstance(element, note.Note):
					notes.append(str(element.pitch))
				elif isinstance(element, chord.Chord):
					notes.append(".".join(str(n) for n in element.normalOrder))

	return notes

all_notes = extract_notes(all_midis)


print("ALL NOTES:", len(all_notes))

count_num = Counter(all_notes)

print("UNIQUE:", len(count_num))

'''
rare_note = []
for index, (key, value) in enumerate(count_num.items()):
    if value < 100:
        m =  key
        rare_note.append(m)
     
print("Total number of notes that occur less than 100 times:", len(rare_note))

for element in all_notes:
    if element in rare_note:
        all_notes.remove(element)

print("Length of Corpus after elemination the rare notes:", len(all_notes))
'''


symb = sorted(list(set(all_notes)))
length_notes = len(all_notes)
length_symb = len(symb)
print(length_symb)
mapping = dict((c, i) for i, c in enumerate(symb))


length = 40
features = []
targets = []
for i in range(0, length_notes - length, 1):
	feature = all_notes[i:i + length]
	target = all_notes[i + length]
	features.append([mapping[j] for j in feature])
	targets.append(mapping[target])

L_datapoints = len(targets)
X = (np.reshape(features, (L_datapoints, length, 1)))/ float(length_symb)
y = tensorflow.keras.utils.to_categorical(targets) 

np.save("all_notes", np.array(all_notes))
np.save("X.npy", X)
np.save("y.npy", y)
print(X.shape)
print(len(y[0]))
print("Saving is done!")

