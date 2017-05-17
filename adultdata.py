import csv
from pandas import *
from matplotlib import *
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
##with open('C:\Users\girishp\Desktop\wine data\winequality-red.csv','r') as f:
##    for line in f:
##     red_wine=line.split()
##        print reddata


##forest_fire = pandas.read_csv("C:\Users\girishp\Desktop\forestfires.csv")
adult_data = pandas.read_csv("C:\Users\girishp\Desktop\adult data\adultdata.csv")

##print red_wine
##red_wine = read_csv('C:\Users\girishp\Desktop\wine data\winequality-red.csv', delimiter=';')

adult_data['randu'] = np.array([np.random.uniform(0,1) for x in range(0, forest_fire.shape[0])])


mp_X_av = adult_data[adult_data.columns[0:12]]
mp_X_train = mp_X_av[adult_data['randu'] <=.67]
mp_X_test = mp_X_av[adult_data['randu'] > .67]


mp_Y_av =adult_data[adult_data.columns[13]]


mp_Y_train = mp_Y_av[adult_data['randu'] <= .67]
mp_Y_test = mp_Y_av[adult_data['randu'] > .67]


rf = RandomForestRegressor(bootstrap=True,
                           criterion='mse', max_depth=2, max_features='auto',
                           min_samples_leaf=1, min_samples_split=2,
                           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
                           verbose=0)
rf.fit(mp_X_train, mp_Y_train)

y_pred = rf.predict(mp_X_test)
print "First model results:"

print 'r squared value of', stats.pearsonr(mp_Y_test, y_pred)[0]**2
print 'RMSE of', sqrt(mean_squared_error(mp_Y_test, y_pred))
