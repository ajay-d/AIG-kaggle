import datetime
import numpy as np
import pandas as pd
import sys
import platform

#from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


########
# DATA #
########

print("Read datasets...")
train_foo = pd.read_pickle("feat3_train_nn.pkl")
test_foo =  pd.read_pickle("feat3_test_nn.pkl")
labels = pd.read_pickle("labels_foo.pkl")
print("Done with datasets...")


# max_depth=4, learning_rate=0.01, n_estimators=4000, colsample_bytree = 0.7,subsample= 0.7
from sklearn.model_selection import StratifiedKFold

clf = XGBClassifier(max_depth=4, learning_rate=0.01, 
                    n_estimators=6000, silent=False, objective='mlogloss', nthread=-1,
                   colsample_bytree = 0.7,subsample= 0.7, ##0.23843 colsam = 0.7  subsample = 0.7
                   colsample_bylevel = 0.7)  #scores_mean 0.23725 without colsample_bylevel


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)


scores = [] 

dataset_blend_train_xgb1 = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
dataset_blend_test_xgb1 = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
i=0
a={}
scores = []

for train_index, test_index in skf.split(train_foo, labels):
        
    print ("Fold ->", i+1)
    X_train, X_test = train_foo.iloc[train_index], train_foo.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='mlogloss', 
          verbose=True,early_stopping_rounds=20)
        
    y_val = clf.predict_proba(X_test)
    
    score = log_loss(y_test, y_val)
    print (score)
    scores.append(score)

    dataset_blend_train_xgb1.iloc[test_index, 0:21] = y_val
        
    a[i] = pd.DataFrame(clf.predict_proba(test_foo)) 
        
    i+=1
        
dataset_blend_test_xgb1 = a[0].copy()
for x in dataset_blend_test_xgb1.columns:
    dataset_blend_test_xgb1[x] = (a[0][x]+a[1][x]+a[2][x]+a[3][x]+a[4][x])/5

scores_mean =  np.mean(scores)
print (scores_mean)

dataset_blend_train_xgb1.to_pickle('FEAT3_LVL1_XGB1_TRAIN_CV_' + str(scores_mean) + '_.pkl')
dataset_blend_test_xgb1.to_pickle('FEAT3_LVL1_XGB1_TEST_CV_'+ str(scores_mean) + '_.pkl')   
    
print ("Done with XGB_1------>")



k = 0

btr= {}  #train
bte= {}   #test
bag_score = []

while k<10:
    
    print ("Iteration->", k+1)
    
    dataset_blend_train_rf2 = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
    dataset_blend_test_rf2 = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
    scores = []
    

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)

    clf= RandomForestClassifier(n_estimators=3000, max_depth = 20, min_samples_leaf = 5, n_jobs=-1, verbose=True,
                           criterion = 'gini', min_impurity_split=1e-08, bootstrap = False)
    
    i=0
    a={}

    for train_index, test_index in skf.split(train_foo, labels):
        
        print ("Fold ->", i+1)
        X_train, X_test = train_foo.iloc[train_index],train_foo.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        clf.fit(X_train, y_train)
        
        y_val = clf.predict_proba(X_test)
    
        score = log_loss(y_test, y_val)
        print (score)
        scores.append(score)

        dataset_blend_train_rf2.iloc[test_index, 0:21] = y_val
        
        a[i] = pd.DataFrame(clf.predict_proba(test_foo))
        
        i+=1
        
    dataset_blend_test_rf2 = a[0].copy()
    for x in dataset_blend_test_rf2.columns:
        dataset_blend_test_rf2[x] = (a[0][x]+a[1][x]+a[2][x]+a[3][x]+a[4][x])/5
    
    scores_mean =  np.mean(scores)
    bag_score.append(scores_mean)
    print (scores_mean)
    
    btr[k] = dataset_blend_train_rf2
    bte[k] = dataset_blend_test_rf2
    
    k+=1

    
dataset_blend_train = btr[0].copy()  #TRAIN bagged
for x in dataset_blend_train.columns:
    dataset_blend_train[x] = (btr[0][x]+btr[1][x]+btr[2][x]+btr[3][x]+btr[4][x]+
                              btr[5][x]+btr[6][x]+btr[7][x]+btr[8][x]+btr[9][x])/10
    
dataset_blend_test = bte[0].copy()  #TEST bagged
for x in dataset_blend_test.columns:
    dataset_blend_test[x] = (bte[0][x]+bte[1][x]+bte[2][x]+bte[3][x]+bte[4][x]+bte[5][x]+bte[6][x]+
                             bte[7][x]+bte[8][x]+bte[9][x])/10
    
dataset_blend_train.to_pickle('FEAT3_LVL1_RF1_BAG10_TRAIN_CV_' + str(np.mean(bag_score))+ '_.pkl')
dataset_blend_test.to_pickle('FEAT3_LVL1_RF1_BAG10_TEST_CV_' + str(np.mean(bag_score)) + '_.pkl')#without econ_unemployment_py
    
print ("Done with bag of bag10 RF ------>")