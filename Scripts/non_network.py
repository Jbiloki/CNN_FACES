# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:34:36 2017

@author: Nguyen
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Scikit Learn support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog

data_frame = pd.read_csv('../Data/train.csv')
data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: list(map(int,x.split(' '))))
#im = Image.fromarray(np.asarray(data_frame['Pixels'].iloc[0]).reshape(48,48))
#im.show()
data_frame['Pixels'] = hog(data_frame['Pixels'])
band = np.array([np.array(band).astype(np.float32) for band in data_frame['Pixels']])
X_train,X_test,y_train,y_test = train_test_split(band, data_frame.Emotion, test_size = 0.33)
model = KNeighborsClassifier()
model.fit(X_train,y_train)
print(accuracy_score(y_test,model.predict(X_test)))