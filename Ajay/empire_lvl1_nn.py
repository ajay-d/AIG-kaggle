import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu2,floatX=float32"
os.environ["KERAS_BACKEND"]= 'theano'
#os.environ["OMP_NUM_THREADS"] = '8'
import theano
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.advanced_activations import PReLU, LeakyReLU
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



train_foo = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
test_foo = pd.read_csv('test_recode_8.csv.gz', compression="gzip")
labels = train_foo['target'].copy()
train_foo.drop(['target', 'claim_id', 'loss'], inplace = True, axis = 1)
test_foo.drop(['claim_id'], inplace = True, axis = 1)


train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

cols2count2log2scale = ['occ_code', 'sic_cd', 'diagnosis_icd9_cd', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
       'cat_6', 'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11', 'cat_12']

for x in cols2count2log2scale:
    ttdata[x].fillna(0, inplace = True)
    #ttdata[x] = ttdata[x].astype(str)
    foo= ttdata[x].value_counts()
    temp = np.log(ttdata[x].apply(lambda x: foo[x]))
    ttdata[x] = temp


cols2scale = ['econ_unemployment_py', 'paid_year', 'econ_gdp_py',
       'econ_gdp_ar_py', 'econ_price_py', 'econ_price_allitems_py',
       'econ_price_healthcare_py', 'econ_10yr_note_py', 'paid_i', 'paid_m',
       'paid_l', 'paid_o', 'any_i', 'any_m', 'any_l', 'any_o', 'mean_i',
       'mean_m', 'mean_l', 'mean_o', 'mean_im', 'mean_all', 'min_i',
       'min_m', 'min_l', 'min_o', 'min_im', 'min_all', 'max_i', 'max_m',
       'max_l', 'max_o', 'max_im', 'max_all', 'med_i', 'med_m', 'med_l',
       'med_o', 'med_im', 'med_all', 'n_hist', 'cutoff_year',
       'avg_wkly_wage', 'imparmt_pct', 'dnb_emptotl', 'dnb_emphere',
       'dnb_worth', 'dnb_sales', 'econ_gdp_ly', 'econ_gdp_ar_ly',
       'econ_price_ly', 'econ_price_allitems_ly',
       'econ_price_healthcare_ly', 'econ_10yr_note_ly',
       'econ_unemployment_ly', 'surgery_yearmo', 'imparmt_pct_yearmo',
       'clmnt_birth_yearmo', 'empl_hire_yearmo', 'death_yearmo',
       'death2_yearmo', 'loss_yearmo', 'reported_yearmo',
       'abstract_yearmo', 'clm_create_yearmo', 'mmi_yearmo',
       'rtrn_to_wrk_yearmo', 'eff_yearmo', 'exp_yearmo', 'suit_yearmo',
       'duration', 'report_to_claim', 'report_lag', 'eff_to_loss',
       'loss_to_mmi', 'loss_to_return', 'years_working', 'age_at_injury',
       'age_at_hire', 'age_at_mmi', 'age_at_return', 'Other', 'PPD', 'PTD',
       'STLMT', 'TTD', 
             
        'occ_code', 'sic_cd', 'diagnosis_icd9_cd', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
       'cat_6', 'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11', 'cat_12' ] 

for x in cols2scale:
    ttdata[x] = scale(ttdata[x])


cols2dummy= ['adj','clmnt_gender', 'major_class_cd', 'prexist_dsblty_in',
       'catas_or_jntcvg_cd', 'suit_matter_type', 'initl_trtmt_cd', 'state',
       'cas_aia_cds_1_2','cas_aia_cds_3_4', 'clm_aia_cds_1_2', 'clm_aia_cds_3_4',
       'law_limit_tt', 'law_limit_pt', 'law_limit_pp', 'law_cola',
       'law_offsets', 'law_ib_scheduled', 'dnb_spevnt_i', 'dnb_rating',
       'dnb_comp_typ']

ttdata = pd.get_dummies(ttdata, columns=cols2dummy)

train_foo = ttdata[ttdata['train_flag']==1]
test_foo = ttdata[ttdata['train_flag']==0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)


#NN bag 10

k = 0

btr= {}  #train
bte= {}   #test

fin_score = []

while k<10:
    
    print ("Iteration->", k+1)
    
    dataset_blend_train_rf2 = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
    dataset_blend_test_rf2 = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
    scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    
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
        #model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Activation('relu'))  #1000->100 0.1dropout - CV10.2424
        #model.add(PReLU())
        model.add(Dense(100, init='normal'))
        model.add(Dropout(0.1))
        model.add(Activation('relu'))     
        #model.add(Dense(50, init='normal'))
        #model.add(Dropout(0.01))
        #model.add(Activation('relu'))     

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
        
        #a[i] = pd.DataFrame(model.predict_proba(blend_test.values)) 
        
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
    dataset_blend_train[x] = (btr[0][x]+btr[1][x]+btr[2][x]+btr[3][x]+btr[4][x]+
                              btr[5][x]+btr[6][x]+btr[7][x]+btr[8][x]+btr[9][x])/10
    
dataset_blend_test = bte[0].copy()  #TEST bagged
for x in dataset_blend_test.columns:
    dataset_blend_test[x] = (bte[0][x]+bte[1][x]+bte[2][x]+bte[3][x]+bte[4][x]+bte[5][x]+bte[6][x]+
                             bte[7][x]+bte[8][x]+bte[9][x])/10
    
#dataset_blend_train.to_pickle('EMPIRE_NN1_RECODE8_TRAIN_' + str(np.mean(fin_score)) +'_.pkl') #nn1
#dataset_blend_test.to_pickle('EMPIRE_NN1_RECODE8_TEST_' + str(np.mean(fin_score))+'_.pkl') # nn1
    
print ("Done with bag of 10 NNs------>") 