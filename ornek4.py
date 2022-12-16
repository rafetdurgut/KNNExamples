

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
filename = "ornek4.csv"
data = pd.read_csv(filename,sep=';')


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=data, x="nem", y="basinc", hue="yagmur")


Y = data['yagmur']
data = data.drop('yagmur',axis=1)
X = data

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)



def objective_function(solution):
    if(sum(solution) == 0):
        return 0
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train.loc[:,solution],Y_train)
    score = model.score(X_test.loc[:,solution], Y_test)
    print(score)
    return score

best_objective = 0;

for i in range(100):
    solution = np.random.random(5)>0.5
    sol_objective = objective_function(solution)    
    if(sol_objective>best_objective):
        best_solution = solution.copy()
        best_objective = sol_objective
        


