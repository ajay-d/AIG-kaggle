import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
os.environ["KERAS_BACKEND"]= 'theano'
#os.environ["OMP_NUM_THREADS"] = '8'
import theano
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical

import datetime
import numpy as np
import pandas as pd
import sys
import platform
import time

from sklearn.preprocessing import LabelEncoder, scale, StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss



print("Read datasets...")
train_foo = pd.read_pickle("feat4_train.pkl")
test_foo = pd.read_pickle("feat4_test.pkl")
labels = pd.read_pickle("labels_foo.pkl")
print("Done with datasets...")

train_foo['imparmt_pct'] = np.log(1+train_foo['imparmt_pct'])
test_foo['imparmt_pct'] = np.log(1+test_foo['imparmt_pct'])



col2scale = ['econ_unemployment_py', 'diagnosis_icd9_cd', 'aww1',
       'imparmt_pct', 'occ_desc1', 'diff_repo_loss', 'diff_impmt_loss',
       'diff_mmi_loss', 'diff_suit_loss', 'diff_paid_repo',
       'diff_paid_impmt', 'diff_paid_mmi', 'diff_paid_death',
       'diff_paid_return', 'diff_paid_suit', 'age_claimaint',
       'target_prev','target_median3', 'i_prev',
       'm_prev', 'tpd_amt', 'l_prev', 'i_prev_avg3', 'm_prev_avg3']



train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

#Try with and without diagnosiscode dummies
col2dummy= ['slice', 'cutoffyr_flag', 'lp1']#'diagnosis_icd9_cd',


ttdata = pd.get_dummies(ttdata, columns=col2dummy)

#ttdata.drop('econ_unemployment_py', inplace = True, axis = 1)
#ttdata.drop('target_median3', inplace = True, axis = 1)
#ttdata.drop('target_prev', inplace = True, axis = 1)

for x in col2scale:
    ttdata[x] = scale(ttdata[x]) #scale
    #ms = MinMaxScaler()           #minmax
    #ttdata[x] = ms.fit_transform(ttdata[x])

train_foo = ttdata[ttdata['train_flag']==1]
test_foo = ttdata[ttdata['train_flag']==0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)




#NN1_LVL1_FEAT3_BAG OF 5 RUNS

k = 0

btr= {}  #train
bte= {}   #test

fin_score = []

while k<5:
    
    print ("Iteration->", k+1)
    
    dataset_blend_train_rf2 = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
    dataset_blend_test_rf2 = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
    scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    
    i=0
    a={}

    for train_index, test_index in skf.split(train_foo, labels):
        
        print ("Fold ->", i+1)
        X_train, X_test = train_foo.iloc[train_index], train_foo.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    
        le_train = LabelEncoder()
        y_train = le_train.fit_transform(y_train)
        y_train = to_categorical(y_train)
    
        le_test = LabelEncoder()
        y_test = le_test.fit_transform(y_test)
        y_test = to_categorical(y_test)
    
        model = Sequential()
        model.add(Dense(1000, input_dim = train_foo.shape[1] , init='normal')) #try init = uniform  510
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, init='normal'))
        model.add(Activation('relu'))     
        model.add(Dropout(0.2))
        model.add(Dense(250, init='normal'))
        model.add(Activation('relu'))     
        model.add(Dropout(0.2))
        model.add(Dense(21, init='normal'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
    
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

        ReduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, 
                                             verbose=2, mode='min',cooldown=5, min_lr=0.00001)

        model.fit(X_train.values, y_train, validation_data = (X_test.values, y_test), 
          batch_size=50, nb_epoch=50,verbose = 2,callbacks=[earlyStopping,ReduceLR])
        
        y_val = model.predict_proba(X_test.values)
    
        score = log_loss(y_test, y_val)
        print (score)
        scores.append(score)

        dataset_blend_train_rf2.iloc[test_index, 0:21] = y_val
        
        a[i] = pd.DataFrame(model.predict_proba(test_foo.values)) 
        
        i+=1
        
    dataset_blend_test_rf2 = a[0].copy()
    for x in dataset_blend_test_rf2.columns:
        dataset_blend_test_rf2[x] = (a[0][x]+a[1][x]+a[2][x]+a[3][x]+a[4][x])/5
    
    scores_mean =  np.mean(scores)
    print (scores_mean)
    fin_score.append(scores_mean)
    
    btr[k] = dataset_blend_train_rf2
    bte[k] = dataset_blend_test_rf2
    
    k+=1


dataset_blend_train = btr[0].copy()  #TRAIN bagged
for x in dataset_blend_train.columns:
    dataset_blend_train[x] = (btr[0][x]+btr[1][x]+btr[2][x]+btr[3][x]+btr[4][x])/5 
                             # btr[5][x]+btr[6][x]+btr[7][x]+btr[8][x]+btr[9][x])/10
    
dataset_blend_test = bte[0].copy()  #TEST bagged
for x in dataset_blend_test.columns:
    dataset_blend_test[x] = (bte[0][x]+bte[1][x]+bte[2][x]+bte[3][x]+bte[4][x])/5 #+bte[5][x]+bte[6][x]+
                             #bte[7][x]+bte[8][x]+bte[9][x])/10
    
dataset_blend_train.to_pickle('NN2_LVL1_FEAT4_BAG5_TRAIN_CV_' + str(np.mean(fin_score)) +'_.pkl') #nn1
dataset_blend_test.to_pickle('NN2_LVL1_FEAT4_BAG5_TEST_CV_' + str(np.mean(fin_score))+'_.pkl') # nn1
    
print ("Done with bag of 5 NNs------>")
