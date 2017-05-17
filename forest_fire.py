import csv
from pandas import *
from matplotlib import *
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
##adata = pd.read_csv("C:\Users\girishp\Desktop/adultdata.csv.tree_3")

forest_fire = pd.read_csv("C:\Users\girishp\Desktop/forestfires.csv")
##fsamples = pd.read_csv("C:\Users\girishp\Desktop\forest_sample\forestfires.binned1.csv")

##print red_wine
##red_wine = read_csv('C:\Users\girishp\Desktop\wine data\winequality-red.csv', delimiter=';')

forest_fire['randu'] = np.array([np.random.uniform(0,1) for x in range(0, forest_fire.shape[0])])

mp_X_av = forest_fire[forest_fire.columns[0:12]]
mp_X_train = mp_X_av[forest_fire['randu'] <=.67]
mp_X_test = mp_X_av[forest_fire['randu'] > .67]
##print mp_X_train 
mp_Y_av =forest_fire[forest_fire.columns[13]]
mp_Y_train = mp_Y_av[forest_fire['randu'] <= .67]
mp_Y_test = mp_Y_av[forest_fire['randu'] > .67]



rf = RandomForestRegressor(bootstrap=True,
                           criterion='mse', max_depth=2, max_features='auto',
                           min_samples_leaf=1, min_samples_split=2,
                           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
                           verbose=0)
rf.fit(mp_X_train, mp_Y_train)

y_pred = rf.predict(mp_X_test)
sns.set_context(context='poster', font_scale=1)
first_test = DataFrame({"Prediction using all predictors" : y_pred, "Actual area predicted" : mp_Y_test})
sns.lmplot(x="Actual area predicted", y="Prediction using all predictors", data=first_test, size=7, aspect=1.5)
sns.plt.title("Forest Fire Data")
sns.plt.show()
print "Prediction on Forest Fire dataset model results:"

print 'R squared value of', stats.pearsonr(mp_Y_test, y_pred)[0]**2
print 'RMSE of', sqrt(mean_squared_error(mp_Y_test, y_pred))

