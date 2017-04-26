import datetime
import numpy as np
import pandas as pd
import sys
import platform
import time

from sklearn.preprocessing import LabelEncoder, scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("Read datasets...")
train_foo = pd.read_pickle("train_foo.pkl")
test_foo = pd.read_pickle("final_test_foo.pkl")
labels = pd.read_pickle("labels_foo.pkl")

print("Done with datasets...")

#SCALING FOR KNN CLASSIFIER
train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

ttdata = scale(ttdata)
ttdata = pd.DataFrame(ttdata, columns = train_foo.columns.values)

train_foo = ttdata[ttdata.train_flag >0]
test_foo = ttdata[ttdata.train_flag <0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)

#KKN_1 K NEAREST NEIGHBORS CLASSIFIER  -  NO DUMMIES, ONLY SCALE  

clf = KNeighborsClassifier(n_neighbors=2000, n_jobs=-1)   

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)


scores = [] 

dataset_blend_train_knn = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
dataset_blend_test_knn = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
i=0
a={}

for train_index, test_index in skf.split(train_foo, labels):
        
    print ("Fold ->", i+1)
    X_train, X_test = train_foo.iloc[train_index], train_foo.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
    clf.fit(X_train, y_train)
        
    y_val = clf.predict_proba(X_test)
    
    score = log_loss(y_test, y_val)
    print (score)
    scores.append(score)

    dataset_blend_train_knn.iloc[test_index, 0:21] = y_val
        
    a[i] = pd.DataFrame(clf.predict_proba(test_foo)) 
        
    i+=1
        
dataset_blend_test_knn = a[0].copy()
for x in dataset_blend_test_knn.columns:
    dataset_blend_test_knn[x] = (a[0][x]+a[1][x]+a[2][x]+a[3][x]+a[4][x])/5

scores_mean =  np.mean(scores)
print (scores_mean)

dataset_blend_train_knn.to_pickle('KNN_1_TRAIN_CV_' + str(scores_mean) + '_.pkl')
dataset_blend_test_knn.to_pickle('KNN_1_TEST_CV_'+ str(scores_mean) + '_.pkl')    
    
print ("Done with KNN------>")