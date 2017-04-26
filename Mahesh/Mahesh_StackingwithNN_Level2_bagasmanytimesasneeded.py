import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu2,floatX=float32"
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
#train_foo = pd.read_pickle("feat3_train_nn.pkl")
#test_foo = pd.read_pickle("feat3_test_nn.pkl")
labels = pd.read_pickle("labels_foo.pkl")
print("Done with datasets...")

#train_foo['imparmt_pct'] = np.log(1+train_foo['imparmt_pct'])
#test_foo['imparmt_pct'] = np.log(1+test_foo['imparmt_pct'])

#data = train_foo.append(test_foo).copy()

#le = LabelEncoder()
#labels = le.fit_transform(labels)
#labels = to_categorical(labels)

#DATASET1 Files   #ADA, ET, KNN, NN, RF2, XGB
feat1_ada1_test = pd.read_pickle('ADA_RF_TEST_0.252704156943_.pkl')
feat1_ada1_train = pd.read_pickle('ADA_RF_TRAIN_0.252704156943_.pkl')
feat1_et1_test = pd.read_pickle('ET_BAG10_TEST_CV_0.244832017963_.pkl')
feat1_et1_train = pd.read_pickle('ET_BAG10_TRAIN_CV_0.244832017963_.pkl')
feat1_knn1_test = pd.read_pickle('KNN_1_TEST_CV_0.292878753905_.pkl')
feat1_knn1_train = pd.read_pickle('KNN_1_TRAIN_CV_0.292878753905_.pkl')
feat1_nn1_test = pd.read_pickle('NN_BAG10_TEST_CV_0.23800_.pkl')
feat1_nn1_train = pd.read_pickle('NN_BAG10_TRAIN_CV_0.23800_.pkl')
feat1_nn2_test = pd.read_pickle('NN_2_BAG10_TEST_CV_0.237944753471_.pkl')
feat1_nn2_train = pd.read_pickle('NN_2_BAG10_TRAIN_CV_0.237944753471_.pkl')
feat1_rf1_test = pd.read_pickle('RF_1_TEST_CV_0.240551549059_.pkl')
feat1_rf1_train = pd.read_pickle('RF_1_TRAIN_CV_0.240551549059_.pkl')
feat1_rf2_test = pd.read_pickle('RF_BAG10_TEST_CV__.pkl')
feat1_rf2_train = pd.read_pickle('RF_BAG10_TRAIN_CV__.pkl')
feat1_xgb1_test = pd.read_pickle('XGB_1_TEST_CV_0.237021859048_.pkl')
feat1_xgb1_train = pd.read_pickle('XGB_1_TRAIN_CV_0.237021859048_.pkl')

for x in feat1_nn1_train.columns.values:
    feat1_nn1_train[x] = (feat1_nn1_train[x] + feat1_nn2_train[x])/2  #BAG NN with latest run
for x in feat1_nn1_test.columns.values:
    feat1_nn1_test[x] = (feat1_nn1_test[x] + feat1_nn2_test[x])/2

#DATASET2 Files   # NN, RF, XGB
feat2_nn1_train = pd.read_pickle('NN1_LVL1_FEAT2_BAG5_TRAIN_CV_0.233310636783_.pkl')
feat2_nn1_test = pd.read_pickle('NN1_LVL1_FEAT2_BAG5_TEST_CV_0.233310636783_.pkl')
feat2_nn2_train = pd.read_pickle('NN2_LVL1_FEAT2_BAG5_TRAIN_CV_0.233326989105_.pkl')
feat2_nn2_test = pd.read_pickle('NN2_LVL1_FEAT2_BAG5_TEST_CV_0.233326989105_.pkl')

feat2_rf1_test = pd.read_pickle('RF1_LVL1_FEAT2_BAG10_TEST_CV_0.238693111279_.pkl')
feat2_rf1_train = pd.read_pickle('RF1_LVL1_FEAT2_BAG10_TRAIN_CV_0.238693111279_.pkl')
feat2_rf2_test = pd.read_pickle('RF2_LVL1_FEAT2_BAG10_TEST_CV_0.238708845758_.pkl')
feat2_rf2_train = pd.read_pickle('RF2_LVL1_FEAT2_BAG10_TRAIN_CV_0.238708845758_.pkl')
feat2_rf3_test = pd.read_pickle('RF3_LVL1_FEAT2_BAG10_TEST_CV_0.238729253353_.pkl')
feat2_rf3_train = pd.read_pickle('RF3_LVL1_FEAT2_BAG10_TRAIN_CV_0.238729253353_.pkl')
feat2_rf4_test = pd.read_pickle('RF4_LVL1_FEAT2_BAG10_TEST_CV_0.23924241984_.pkl')
feat2_rf4_train = pd.read_pickle('RF4_LVL1_FEAT2_BAG10_TRAIN_CV_0.23924241984_.pkl')

feat2_xgb1_test = pd.read_pickle('XGB1_lvl1_feat2_TEST_CV_0.235280296454_.pkl')
feat2_xgb1_train = pd.read_pickle('XGB1_lvl1_feat2_TRAIN_CV_0.235280296454_.pkl')

#NN bag
for x in feat2_nn1_train.columns.values:
    feat2_nn1_train[x] = (feat2_nn1_train[x] + feat2_nn2_train[x])/2  #BAG NN with latest run
for x in feat2_nn1_test.columns.values:
    feat2_nn1_test[x] = (feat2_nn1_test[x] + feat2_nn2_test[x])/2

#RF bag    
for x in feat2_rf1_train.columns.values:
    feat2_rf1_train[x] = (feat2_rf1_train[x] + feat2_rf2_train[x] +feat2_rf3_train[x] +feat2_rf4_train[x])/4 
for x in feat2_rf1_test.columns.values:
    feat2_rf1_test[x] = (feat2_rf1_test[x] + feat2_rf2_test[x] + feat2_rf3_test[x]   +feat2_rf4_test[x])/4
    

#DATASET3 Files   #LR, NN, XGB, RF
feat3_lr_test = pd.read_pickle('FEAT3_LVL1_LR_TEST_0.261997606724_.pkl')
feat3_lr_train = pd.read_pickle('FEAT3_LVL1_LR_TRAIN_0.261997606724_.pkl')

feat3_nn4_test = pd.read_pickle('FEAT3_LVL1_NN2BAG5_TEST_0.235852318051_.pkl')
feat3_nn4_train = pd.read_pickle('FEAT3_LVL1_NN2BAG5_TRAIN_0.235852318051_.pkl')
feat3_nn1_test = pd.read_pickle('FEAT3_LVL1_NN2BAG5_TEST_0.23571687975_.pkl')
feat3_nn1_train = pd.read_pickle('FEAT3_LVL1_NN2BAG5_TRAIN_0.23571687975_.pkl')
feat3_nn2_test = pd.read_pickle('NN1_LVL1_FEAT3_BAG5_TEST_CV_0.235682233291_.pkl')
feat3_nn2_train = pd.read_pickle('NN1_LVL1_FEAT3_BAG5_TRAIN_CV_0.235682233291_.pkl')
feat3_nn3_test = pd.read_pickle('NN1_LVL1_FEAT3_BAG5_TEST_CV_0.235798564411_.pkl')
feat3_nn3_train = pd.read_pickle('NN1_LVL1_FEAT3_BAG5_TRAIN_CV_0.235798564411_.pkl')

feat3_xgb1_test = pd.read_pickle('FEAT3_LVL1_XGB1_TEST_CV_0.235067276521_.pkl')
feat3_xgb1_train = pd.read_pickle('FEAT3_LVL1_XGB1_TRAIN_CV_0.235067276521_.pkl')
feat3_rf1_test = pd.read_pickle('FEAT3_LVL1_RF1_BAG10_TEST_CV_0.237585230498_.pkl')
feat3_rf1_train = pd.read_pickle('FEAT3_LVL1_RF1_BAG10_TRAIN_CV_0.237585230498_.pkl')

#lr, nn, xgb, rf

#NN bag
for x in feat3_nn1_train.columns.values:
    feat3_nn1_train[x] = (feat3_nn1_train[x] + feat3_nn2_train[x] + feat3_nn3_train[x]+ feat3_nn4_train[x])/4
for x in feat3_nn1_test.columns.values:
    feat3_nn1_test[x] = (feat3_nn1_test[x] + feat3_nn2_test[x] + feat3_nn3_test[x] + feat3_nn4_test[x])/4


#DATASET4 Files   # LR, NN, RF, XGB
feat4_lr_test = pd.read_pickle('FEAT4_LVL1_LR_TEST_0.263706807529_.pkl')
feat4_lr_train = pd.read_pickle('FEAT4_LVL1_LR_TRAIN_0.263706807529_.pkl')

feat4_nn1_test = pd.read_pickle('NN2_LVL1_FEAT4_BAG5_TEST_CV_0.235438827759_.pkl')
feat4_nn1_train = pd.read_pickle('NN2_LVL1_FEAT4_BAG5_TRAIN_CV_0.235438827759_.pkl')
feat4_nn2_test = pd.read_pickle('NN2_LVL1_FEAT4_BAG5_TEST_CV_0.235470699625_.pkl')
feat4_nn2_train = pd.read_pickle('NN2_LVL1_FEAT4_BAG5_TRAIN_CV_0.235470699625_.pkl')

feat4_rf1_test = pd.read_pickle('FEAT4_LVL1_RF1_BAG10_TEST_CV_0.237607150276_.pkl')
feat4_rf1_train = pd.read_pickle('FEAT4_LVL1_RF1_BAG10_TRAIN_CV_0.237607150276_.pkl')
feat4_xgb1_test = pd.read_pickle('FEAT4_LVL1_XGB1_TEST_CV_0.234934148929_.pkl')
feat4_xgb1_train = pd.read_pickle('FEAT4_LVL1_XGB1_TRAIN_CV_0.234934148929_.pkl')


for x in feat4_nn1_train.columns.values:
    feat4_nn1_train[x] = (feat4_nn1_train[x] + feat4_nn2_train[x])/2
for x in feat4_nn1_test.columns.values:
    feat4_nn1_test[x] = (feat4_nn1_test[x] + feat4_nn2_test[x])/2


blend_train = pd.concat([feat1_ada1_train, feat1_et1_train, feat1_knn1_train , feat1_nn1_train, feat1_rf2_train, 
                         feat1_xgb1_train, 
                         feat2_nn1_train, feat2_rf1_train, feat2_xgb1_train,
                         feat3_lr_train, feat3_nn1_train, feat3_rf1_train, feat3_xgb1_train,
                         feat4_lr_train, feat4_nn1_train, feat4_rf1_train, feat4_xgb1_train],
                         axis=1, ignore_index=True) #adding nn2

#blend_train = blend_train.round(3)

blend_test = pd.concat([feat1_ada1_test, feat1_et1_test, feat1_knn1_test , feat1_nn1_test, feat1_rf2_test, 
                         feat1_xgb1_test, 
                         feat2_nn1_test, feat2_rf1_test, feat2_xgb1_test,
                         feat3_lr_test, feat3_nn1_test, feat3_rf1_test, feat3_xgb1_test,
                         feat4_lr_test, feat4_nn1_test, feat4_rf1_test, feat4_xgb1_test], 
                         axis=1, ignore_index = True)


blend_test.shape


#Increasing the number of runs (k) improves score i.e. averaging more runs of Neural Nets keeps increasing the score

k = 0

btr= {}  #train
bte= {}   #test

fin_score = []

while k<5:
    
    print ("Iteration->", k+1)
    
    dataset_blend_train_rf2 = pd.DataFrame(np.zeros((blend_train.shape[0], 21)))
    dataset_blend_test_rf2 = pd.DataFrame(np.zeros((blend_test.shape[0], 21)))
    
    scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
    
    i=0
    a={}

    for train_index, test_index in skf.split(blend_train, labels):
        
        print ("Fold ->", i+1)
        X_train, X_test = blend_train.iloc[train_index], blend_train.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    
        le_train = LabelEncoder()
        y_train = le_train.fit_transform(y_train)
        y_train = to_categorical(y_train)
    
        le_test = LabelEncoder()
        y_test = le_test.fit_transform(y_test)
        y_test = to_categorical(y_test)
    
        model = Sequential()
        model.add(Dense(400, input_dim = blend_train.shape[1] , init='normal')) 
        #model.add(BatchNormalization())
        model.add(Activation('relu'))  #400 - dropout 0.2 #CV 0.225590144636
        model.add(Dropout(0.2))
        model.add(Dense(21, init='normal'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    
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
        
        a[i] = pd.DataFrame(model.predict_proba(blend_test.values)) 
        
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
    
dataset_blend_train.to_pickle('STACK_LVL2_NNadam_METAONLY_BAG5_400_TRAIN_CV_' + str(np.mean(fin_score)) +'_.pkl') #nn1
dataset_blend_test.to_pickle('STACK_LVL2_NNadam_METAONLY_BAG5_400_TEST_CV_' + str(np.mean(fin_score))+'_.pkl') # nn1
    
print ("Done with bag of 5 NNs------>")