# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:55:13 2021

@author: vibis
"""

import itertools
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
import matplotlib.pyplot as plt
import pickle   
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings("ignore")

'''
Since code is messy little explanation so basically have the 
data folder in same directory change name of folder in code accordingly
block 1 down in the code for training and dumping in pickle file
block 2 for opening trained weights from pickle file
block 3 for testing on whatever accordingly see
'''
directory = "C://Users/vibis/Downloads/Dataset_ Dig Rec System- 15 occ/"
#directory = "Complete Dataset/"

def build_dataset(sound_path=directory):
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = dict()
    n = len(files)
    for i in tqdm(range(n)):
        feature = feature_extractor(sound_path=sound_path + files[i])
        digit = files[i][0]
        if digit not in data.keys():
            data[digit] = []
            x_test.append(feature)
            y_test.append(digit)
        else:
            if np.random.rand() < 0.1:
                x_test.append(feature)
                y_test.append(digit)
            else:
                x_train.append(feature)
                y_train.append(digit)
            data[digit].append(feature)
    return x_train, y_train, x_test, y_test, data


def feature_extractor(sound_path):
    sampling_freq, audio = wavfile.read(sound_path)
    mfcc_features = mfcc(audio, sampling_freq,nfft = 1200,numcep=39,nfilt=40)
    return mfcc_features


def train_model(data):
    learned_hmm = dict()
    for label in data.keys():
        model = hmm.GMMHMM(verbose=False,n_components=10,n_iter=100)
        feature = np.ndarray(shape=(1, 39))
        for list_feature in tqdm(data[label]):
            feature = np.vstack((feature, list_feature))

        # TIME CONSUMING PART    
        obj = model.fit(feature)
        learned_hmm[label] = obj
    return learned_hmm


def prediction(test_data, trained):
    # predict list of test
    predict_label = []
    names = []
    if type(test_data) == type([]):
        for test in test_data:
            scores = []
            for node in trained.keys():
                scores.append(trained[node].score(test))
                names.append(node)
            predict_label.append(scores.index(max(scores)))
    else:
        scores = []
        for node in trained.keys():
            scores.append(trained[node].score(test_data))
            names.append(node)
        predict_label.append(scores.index(max(scores)))
    return names[predict_label[0]]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(f'confmat_{random.randint(1,100)}.png')


def report(y_test, y_pred, show_cm=True):
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    if show_cm:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

'''Block 1 start'''
print("BUILDING DATASET")
x_train, y_train, x_test, y_test, data = build_dataset()

print("STARTING")
learned_hmm = train_model(data)
with open("learned_tel3.pkl", "wb") as file:
    pickle.dump(learned_hmm, file)
print("training done")
'''Block 1 end'''

'''Block 2 start'''
with open("learned_tel3.pkl", "rb") as file:
    learned_hmm = pickle.load(file)
'''Block 2 start'''


'''Block 3 start'''
boros = directory
files = sorted(os.listdir(directory))
tot_test = 0
tot_train = 0
n = len(x_test)
m = len(x_train)
pred_test = []
pred_train = []
for i in range(n):
    y_pred = prediction(x_test[i], learned_hmm)
  

    if y_pred == y_test[i]:
        tot_test += 1
    pred_test.append(y_pred)
for i in range(m):
    y_pred = prediction(x_train[i], learned_hmm)
    if y_pred == y_train[i]:
        tot_train += 1
    pred_train.append(y_pred)

report(y_test,pred_test)
report(y_train,pred_train)
print('########################## TRAINING ACCURACY ######################################')
print(tot_train/m)
print('########################## TESTING ACCURACY #######################################')
print(tot_test/n)

'''Block 3 end'''