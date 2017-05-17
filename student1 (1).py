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
mat_perf = read_csv('C:\Users\girishp\Desktop\student\student-mat.csv', delimiter=';')
mat_sample_perf = read_csv('C:\Users\girishp\Desktop\student\sample\student-mat.binned.g1_1.csv', delimiter=';')


#Measure of importance among variables

test_stats = {'variable': [],'test_type': [],'test_value': []}

for col in mat_perf.columns[:-3]:
    test_stats['variable'].append(col)
    if mat_perf[col].dtype == 'O':
        # Do Anova
##        print col + "(python object)" +'\n' 
        aov = smf.ols(formula='G3 ~ '+ col, data=mat_perf, missing='drop').fit()
        test_stats['test_type'].append('F Test')
        test_stats['test_value'].append(round(aov.fvalue,2))
    else:
        #do correlation
        print col + '\n'
        model = smf.ols(formula='G3 ~ '+ col, data=mat_perf, missing='drop').fit()
        value = round(model.tvalues[1],2)
        test_stats['test_type'].append('t Test')
        test_stats['test_value'].append(value)

test_stats = DataFrame(test_stats)
test_stats.sort(columns='test_value', ascending=False, inplace=True)

##print(test_stats)

##Plotting the graphs
f, (ax1, ax2) = plt.subplots(2,1, sharex=False)
sns.set_context(context='poster', font_scale=1)
sns.barplot(x='variable', y='test_value', data=test_stats.query("test_type == 'F Test' "), hline=.1, ax=ax1, x_order=[x for x in test_stats.query("test_type == 'F Test'")['variable']])
ax1.set_ylabel('F values')
ax1.set_xlabel('')

sns.barplot(x='variable', y='test_value', data=test_stats.query("test_type == 't Test' "), hline=.1, ax=ax2, x_order=[x for x in test_stats.query("test_type == 't Test'")['variable']])
ax2.set_ylabel('t Values')
ax2.set_xlabel('')

sns.despine(bottom=True)
plt.tight_layout(h_pad=3)
##plt.show()



#Training Firstrandommodel

usevars = [x for x in test_stats.query("test_value >= 3.0 | test_value <=-3.0")['variable']]
mat_perf['randu'] = np.array([np.random.uniform(0,1) for x in range(0, mat_perf.shape[0])])
mp_X = mat_perf[usevars]
mp_X_train = mp_X[mat_perf['randu'] <=.67]
mp_X_test = mp_X[mat_perf['randu'] > .67]

mp_Y_train_G1 = mat_perf.G1[mat_perf['randu']<=.67]
mp_Y_test_G1 = mat_perf.G1[mat_perf['randu'] >.67]

mp_Y_train_G2 = mat_perf.G2[mat_perf['randu']<=.67]
mp_Y_test_G2 = mat_perf.G2[mat_perf['randu'] >.67]

mp_Y_train_G3 = mat_perf.G3[mat_perf['randu']<=.67]
mp_Y_test_G3 = mat_perf.G3[mat_perf['randu']> .67]

#for training set
cat_cols = [x for x in mp_X_train.columns if mp_X_train[x].dtype == 'O']
for col in cat_cols:
    new_cols = get_dummies(mp_X_train[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_train = concat([mp_X_train, new_cols], axis=1)

#for testing set
cat_cols = [x for x in mp_X_test.columns if mp_X_test[x].dtype == 'O']
for col in cat_cols:
    new_cols = get_dummies(mp_X_test[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_test = concat([mp_X_test, new_cols], axis=1)
##print(mp_X_test)
##print(mp_Y_train)
mp_X_train.drop(cat_cols, inplace=True, axis=1)
mp_X_test.drop(cat_cols, inplace=True, axis=1)

rf = RandomForestRegressor(bootstrap=True,
                           criterion='mse', max_depth=2, max_features='auto',
                           min_samples_leaf=1, min_samples_split=2,
                           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
                           verbose=0)
rf.fit(mp_X_train, mp_Y_train_G1)

##testing the first model
y_pred = rf.predict(mp_X_test)
sns.set_context(context='poster', font_scale=1)
first_test = DataFrame({"pred.G1.keepvars" : y_pred, "G1" : mp_Y_test_G1})
sns.lmplot(x="G1", y="pred.G1.keepvars", data=first_test, size=7, aspect=1.5)
##sns.lmplot("G3","pred.G3.keepvars", first_test, size=7, aspect=1.5)

print "First model results using only important variables using variable importance:"

print 'r squared value of', stats.pearsonr(mp_Y_test_G1, y_pred)[0]**2
print 'RMSE of', sqrt(mean_squared_error(mp_Y_test_G1, y_pred))
##sns.plt.show()


#print the variable importances generated for each training model

importances = DataFrame({'cols':mp_X_train.columns, 'imps':rf.feature_importances_})
print importances.sort(['imps'], ascending=False)







###Training Second RandomModel
##
##
##av = almost all variables
mp_X_av = mat_perf[mat_perf.columns[0:30]]
mp_X_train_av = mp_X_av[mat_perf['randu'] <= .67]
mp_X_test_av = mp_X_av[mat_perf['randu']> .67]
##
###for the training set
cat_cols = [x for x in mp_X_train_av.columns if mp_X_train_av[x].dtype == "O"]
for col in cat_cols:
    new_cols = get_dummies(mp_X_train_av[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_train_av = concat([mp_X_train_av, new_cols], axis=1)
    
##
cat_cols = [x for x in mp_X_test_av.columns if mp_X_test_av[x].dtype == "O"]
for col in cat_cols:
    new_cols = get_dummies(mp_X_test_av[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_test_av = concat([mp_X_test_av, new_cols], axis=1)
##
mp_X_train_av.drop(cat_cols, inplace=True, axis=1)
mp_X_test_av.drop(cat_cols, inplace=True, axis=1)
##
rf_av = RandomForestRegressor(bootstrap=True,
                           criterion='mse', max_depth=2, max_features='auto',
                           min_samples_leaf=1, min_samples_split=2,
                           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
                           verbose=0)
rf_av.fit(mp_X_train_av, mp_Y_train_G3)
##
##
y_pred_av = rf_av.predict(mp_X_test_av)
second_test = DataFrame({"Predictions made using G3" : y_pred_av, "Actual G3" : mp_Y_test_G1})
sns.lmplot(x="Actual G3", y="Predictions made using G3", data=second_test, size=7, aspect=1.5)
##sns.lmplot("G3","pred.G3.keepvars", first_test, size=7, aspect=1.5)

print "Main answer!"
print "Second model results using all predictors:"

print 'R Squared value of', stats.pearsonr(mp_Y_test_G3, y_pred_av)[0]**2
print 'MSE of', sqrt(mean_squared_error(mp_Y_test_G3, y_pred_av))
sns.plt.show()



importances_av = DataFrame({'cols':mp_X_train_av.columns, 'imps':rf_av.feature_importances_})
print importances_av.sort(['imps'], ascending=False)
