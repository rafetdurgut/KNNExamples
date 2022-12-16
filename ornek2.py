# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:02:01 2022

@author: rdurgut
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:21:27 2022

@author: rdurgut
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

filename = "ornek2.csv"
data = pd.read_csv(filename,sep=';')

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=data, x="nem", y="basinc", hue="yagmur")

Y = data['yagmur']
data = data.drop('yagmur',axis=1)
X = data

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)


model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,Y_train)


correct_predictions = model.score(X_test, Y_test)







