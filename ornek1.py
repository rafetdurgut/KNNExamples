# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:21:27 2022

@author: rdurgut
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


filename = "ornek1.csv"
data = pd.read_csv(filename,sep=';')


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=data, x="nem", y="basinc", hue="yagmur")


Y = data['yagmur']
data = data.drop('yagmur',axis=1)
X = data

test_X = [ [100,650] ]

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X,Y)

plt.scatter(test_X[0][0], test_X[0][1],c='r')
predicted_output = model.predict(test_X)





