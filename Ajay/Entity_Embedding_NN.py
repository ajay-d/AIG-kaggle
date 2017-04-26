import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["KERAS_BACKEND"]= 'theano'
#os.environ["OMP_NUM_THREADS"] = '8'
#import theano
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


########
# DATA #
########
train_foo = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test_foo = pd.read_csv('test_recode_12.csv.gz', compression="gzip")
labels = train_foo['target'].copy()
train_foo.drop(['target', 'claim_id', 'loss'], inplace = True, axis = 1)
test_foo.drop(['claim_id'], inplace = True, axis = 1)


train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

#(ttdata.apply(pd.Series.nunique))
#(ttdata[col2trunc].apply(pd.Series.nunique))

col2scale = ['econ_unemployment_py', 'paid_year', 'econ_gdp_py',
       'econ_gdp_ar_py', 'econ_price_py', 'econ_price_allitems_py',
       'econ_price_healthcare_py', 'econ_10yr_note_py', 'paid_i', 'paid_m',
       'paid_l', 'paid_o', 'pct_paid_i', 'pct_paid_m', 'pct_paid_im',
       'pct_paid_i_scale', 'pct_paid_m_scale', 'pct_paid_im_scale',
       'pct_paid_i_scale_sq', 'pct_paid_m_scale_sq',
       'pct_paid_im_scale_sq', 'pct_paid_i_scale_dollar',
       'pct_paid_m_scale_dollar', 'pct_paid_im_scale_dollar',
       'pct_paid_i_scale_sq_dollar', 'pct_paid_m_scale_sq_dollar',
       'pct_paid_im_scale_sq_dollar', 'any_i', 'any_m', 'any_l', 'any_o',
       'mean_i', 'mean_m', 'mean_l', 'mean_o', 'mean_im', 'mean_all',
       'sd_i', 'sd_m', 'sd_l', 'sd_o', 'sd_im', 'sd_all', 'min_i', 'min_m',
       'min_l', 'min_o', 'min_im', 'min_all', 'max_i', 'max_m', 'max_l',
       'max_o', 'max_im', 'max_all', 'med_i', 'med_m', 'med_l', 'med_o',
       'med_im', 'med_all', 'n_hist', 'pct_paid_i_ave', 'pct_paid_m_ave',
       'pct_paid_im_ave', 'pct_paid_i_ave_scale', 'pct_paid_m_ave_scale',
       'pct_paid_im_ave_scale', 'pct_paid_i_ave_scale_sq',
       'pct_paid_m_ave_scale_sq', 'pct_paid_im_ave_scale_sq',
       'pct_paid_i_ave_scale_dollar', 'pct_paid_m_ave_scale_dollar',
       'pct_paid_im_ave_scale_dollar', 'pct_paid_i_ave_scale_sq_dollar',
       'pct_paid_m_ave_scale_sq_dollar', 'pct_paid_im_ave_scale_sq_dollar',
       'cutoff_year', 'avg_wkly_wage', 'imparmt_pct', 'dnb_emptotl',
       'dnb_emphere', 'dnb_worth', 'dnb_sales', 'econ_gdp_ly',
       'econ_gdp_ar_ly', 'econ_price_ly', 'econ_price_allitems_ly',
       'econ_price_healthcare_ly', 'econ_10yr_note_ly',
       'econ_unemployment_ly', 'surgery_yearmo', 'imparmt_pct_yearmo',
       'clmnt_birth_yearmo', 'empl_hire_yearmo', 'death_yearmo',
       'death2_yearmo', 'loss_yearmo', 'reported_yearmo',
       'abstract_yearmo', 'clm_create_yearmo', 'mmi_yearmo',
       'rtrn_to_wrk_yearmo', 'eff_yearmo', 'exp_yearmo', 'suit_yearmo',
       'duration', 'report_to_claim', 'report_lag', 'eff_to_loss',
       'loss_to_mmi', 'loss_to_return', 'years_working', 'paid_report',
       'age_at_injury', 'age_at_hire', 'age_at_mmi', 'age_at_return',
       'any_sic', 'any_icd', 'paid_il1', 'paid_ml1', 'paid_il2',
       'paid_ml2', 'paid_il3', 'paid_ml3', 'Other', 'PPD', 'PTD', 'STLMT',
       'TTD', 'mc_code_I01', 'mc_code_I02', 'mc_code_I03', 'mc_code_I04',
       'mc_code_I05', 'mc_code_I06', 'mc_code_I07', 'mc_code_I11',
       'mc_code_I12', 'mc_code_I14', 'mc_code_I15', 'mc_code_I17',
       'mc_code_I18', 'mc_code_I19', 'mc_code_I6I', 'mc_code_II1',
       'mc_code_II3', 'mc_code_II6', 'mc_code_II8', 'mc_code_II9',
       'mc_code_IIH', 'mc_code_IIU', 'mc_code_IS1', 'mc_code_IT9',
       'mc_code_IXI', 'mc_code_L', 'mc_code_L00', 'mc_code_L24',
       'mc_code_L2M', 'mc_code_L56', 'mc_code_L57', 'mc_code_L58',
       'mc_code_L59', 'mc_code_L60', 'mc_code_L61', 'mc_code_L62',
       'mc_code_L63', 'mc_code_L64', 'mc_code_L66', 'mc_code_L72',
       'mc_code_L73', 'mc_code_L75', 'mc_code_L79', 'mc_code_LCT',
       'mc_code_LDL', 'mc_code_LF2', 'mc_code_LHR', 'mc_code_LL1',
       'mc_code_LL2', 'mc_code_LL4', 'mc_code_LL6', 'mc_code_LL9',
       'mc_code_LLB', 'mc_code_LLD', 'mc_code_LLH', 'mc_code_LLI',
       'mc_code_LNN', 'mc_code_LPC', 'mc_code_LPW', 'mc_code_LU7',
       'mc_code_LXP', 'mc_code_M1A', 'mc_code_M21', 'mc_code_M23',
       'mc_code_M25', 'mc_code_M26', 'mc_code_M27', 'mc_code_M28',
       'mc_code_M29', 'mc_code_M31', 'mc_code_M32', 'mc_code_M33',
       'mc_code_M34', 'mc_code_M40', 'mc_code_M4M', 'mc_code_MBA',
       'mc_code_MCA', 'mc_code_MCL', 'mc_code_MCS', 'mc_code_MDA',
       'mc_code_MFC', 'mc_code_MHO', 'mc_code_MM1', 'mc_code_MM2',
       'mc_code_MM3', 'mc_code_MM6', 'mc_code_MMO', 'mc_code_MMQ',
       'mc_code_MMS', 'mc_code_MMU', 'mc_code_MMV', 'mc_code_MMW',
       'mc_code_MP8', 'mc_code_MPS', 'mc_code_MS4', 'mc_code_MSI',
       'mc_code_MSO', 'mc_code_MTC', 'mc_code_MTF', 'paid_im', 'reg_i',
       'reg_m', 'reg_im', 'reg_i_scale', 'reg_m_scale', 'reg_im_scale',
       'reg_i_scale_sq', 'reg_m_scale_sq', 'reg_im_scale_sq',
       'paid_im_scale1', 'paid_im_scale2', 'paid_im_scale1_sq',
       'paid_im_scale2_sq']

for x in col2scale:
    ms = MinMaxScaler()
    ttdata[x] = ms.fit_transform(ttdata[x])

col2cat = ['cat_slice', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
       'cat_6', 'cat_7', 'cat_8', 
       'cat_13', 'cat_14', 'cat_15', 'cat_16', 'cat_17', 'cat_18',
       'cat_19', 'cat_20', 'cat_21', 'cat_22', 'cat_23', 'cat_24',
        'cat_28',  'cat_33', 'cat_34', 'cat_35', 'cat_36',
       'cat_37', 'cat_38', 'cat_39', 'cat_40', 'cat_41', 'cat_42', 'cat_9', 'cat_10', 'cat_11', 'cat_12','cat_25', 'cat_26', 'cat_27', 'cat_29' ,'cat_30', 'cat_31' ,'cat_32',
                'cat_43', 'cat_44','cat_45' ,'cat_46','cat_47', 'cat_48',
           'cat_49', 'cat_50', 'cat_51', 'cat_52', 'cat_53', 'cat_54',
           'cat_55', 'cat_56', 'cat_57', 'cat_58','cat_59','cat_60', 'cat_61']


#ttdata = pd.get_dummies(ttdata, columns=col2cat, sparse = True)


train_foo = ttdata[ttdata['train_flag']==1]
test_foo = ttdata[ttdata['train_flag']==0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)



allcols = ['econ_unemployment_py', 'paid_year', 'econ_gdp_py',
       'econ_gdp_ar_py', 'econ_price_py', 'econ_price_allitems_py',
       'econ_price_healthcare_py', 'econ_10yr_note_py', 'paid_i', 'paid_m',
       'paid_l', 'paid_o', 'pct_paid_i', 'pct_paid_m', 'pct_paid_im',
       'pct_paid_i_scale', 'pct_paid_m_scale', 'pct_paid_im_scale',
       'pct_paid_i_scale_sq', 'pct_paid_m_scale_sq',
       'pct_paid_im_scale_sq', 'pct_paid_i_scale_dollar',
       'pct_paid_m_scale_dollar', 'pct_paid_im_scale_dollar',
       'pct_paid_i_scale_sq_dollar', 'pct_paid_m_scale_sq_dollar',
       'pct_paid_im_scale_sq_dollar', 'any_i', 'any_m', 'any_l', 'any_o',
       'mean_i', 'mean_m', 'mean_l', 'mean_o', 'mean_im', 'mean_all',
       'sd_i', 'sd_m', 'sd_l', 'sd_o', 'sd_im', 'sd_all', 'min_i', 'min_m',
       'min_l', 'min_o', 'min_im', 'min_all', 'max_i', 'max_m', 'max_l',
       'max_o', 'max_im', 'max_all', 'med_i', 'med_m', 'med_l', 'med_o',
       'med_im', 'med_all', 'n_hist', 'pct_paid_i_ave', 'pct_paid_m_ave',
       'pct_paid_im_ave', 'pct_paid_i_ave_scale', 'pct_paid_m_ave_scale',
       'pct_paid_im_ave_scale', 'pct_paid_i_ave_scale_sq',
       'pct_paid_m_ave_scale_sq', 'pct_paid_im_ave_scale_sq',
       'pct_paid_i_ave_scale_dollar', 'pct_paid_m_ave_scale_dollar',
       'pct_paid_im_ave_scale_dollar', 'pct_paid_i_ave_scale_sq_dollar',
       'pct_paid_m_ave_scale_sq_dollar', 'pct_paid_im_ave_scale_sq_dollar',
       'cutoff_year', 'avg_wkly_wage', 'imparmt_pct', 'dnb_emptotl',
       'dnb_emphere', 'dnb_worth', 'dnb_sales', 'econ_gdp_ly',
       'econ_gdp_ar_ly', 'econ_price_ly', 'econ_price_allitems_ly',
       'econ_price_healthcare_ly', 'econ_10yr_note_ly',
       'econ_unemployment_ly', 'surgery_yearmo', 'imparmt_pct_yearmo',
       'clmnt_birth_yearmo', 'empl_hire_yearmo', 'death_yearmo',
       'death2_yearmo', 'loss_yearmo', 'reported_yearmo',
       'abstract_yearmo', 'clm_create_yearmo', 'mmi_yearmo',
       'rtrn_to_wrk_yearmo', 'eff_yearmo', 'exp_yearmo', 'suit_yearmo',
       'duration', 'report_to_claim', 'report_lag', 'eff_to_loss',
       'loss_to_mmi', 'loss_to_return', 'years_working', 'paid_report',
       'age_at_injury', 'age_at_hire', 'age_at_mmi', 'age_at_return',
       'any_sic', 'any_icd', 'paid_il1', 'paid_ml1', 'paid_il2',
       'paid_ml2', 'paid_il3', 'paid_ml3', 'Other', 'PPD', 'PTD', 'STLMT',
       'TTD', 'mc_code_I01', 'mc_code_I02', 'mc_code_I03', 'mc_code_I04',
       'mc_code_I05', 'mc_code_I06', 'mc_code_I07', 'mc_code_I11',
       'mc_code_I12', 'mc_code_I14', 'mc_code_I15', 'mc_code_I17',
       'mc_code_I18', 'mc_code_I19', 'mc_code_I6I', 'mc_code_II1',
       'mc_code_II3', 'mc_code_II6', 'mc_code_II8', 'mc_code_II9',
       'mc_code_IIH', 'mc_code_IIU', 'mc_code_IS1', 'mc_code_IT9',
       'mc_code_IXI', 'mc_code_L', 'mc_code_L00', 'mc_code_L24',
       'mc_code_L2M', 'mc_code_L56', 'mc_code_L57', 'mc_code_L58',
       'mc_code_L59', 'mc_code_L60', 'mc_code_L61', 'mc_code_L62',
       'mc_code_L63', 'mc_code_L64', 'mc_code_L66', 'mc_code_L72',
       'mc_code_L73', 'mc_code_L75', 'mc_code_L79', 'mc_code_LCT',
       'mc_code_LDL', 'mc_code_LF2', 'mc_code_LHR', 'mc_code_LL1',
       'mc_code_LL2', 'mc_code_LL4', 'mc_code_LL6', 'mc_code_LL9',
       'mc_code_LLB', 'mc_code_LLD', 'mc_code_LLH', 'mc_code_LLI',
       'mc_code_LNN', 'mc_code_LPC', 'mc_code_LPW', 'mc_code_LU7',
       'mc_code_LXP', 'mc_code_M1A', 'mc_code_M21', 'mc_code_M23',
       'mc_code_M25', 'mc_code_M26', 'mc_code_M27', 'mc_code_M28',
       'mc_code_M29', 'mc_code_M31', 'mc_code_M32', 'mc_code_M33',
       'mc_code_M34', 'mc_code_M40', 'mc_code_M4M', 'mc_code_MBA',
       'mc_code_MCA', 'mc_code_MCL', 'mc_code_MCS', 'mc_code_MDA',
       'mc_code_MFC', 'mc_code_MHO', 'mc_code_MM1', 'mc_code_MM2',
       'mc_code_MM3', 'mc_code_MM6', 'mc_code_MMO', 'mc_code_MMQ',
       'mc_code_MMS', 'mc_code_MMU', 'mc_code_MMV', 'mc_code_MMW',
       'mc_code_MP8', 'mc_code_MPS', 'mc_code_MS4', 'mc_code_MSI',
       'mc_code_MSO', 'mc_code_MTC', 'mc_code_MTF', 'paid_im', 'reg_i',
       'reg_m', 'reg_im', 'reg_i_scale', 'reg_m_scale', 'reg_im_scale',
       'reg_i_scale_sq', 'reg_m_scale_sq', 'reg_im_scale_sq',
       'paid_im_scale1', 'paid_im_scale2', 'paid_im_scale1_sq',
       'paid_im_scale2_sq', 
        'cat_slice', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
       'cat_6', 'cat_7', 'cat_8', 
       'cat_13', 'cat_14', 'cat_15', 'cat_16', 'cat_17', 'cat_18',
       'cat_19', 'cat_20', 'cat_21', 'cat_22', 'cat_23', 'cat_24',
        'cat_28',  'cat_33', 'cat_34', 'cat_35', 'cat_36',
       'cat_37', 'cat_38', 'cat_39', 'cat_40', 'cat_41', 'cat_42', 'cat_9', 'cat_10', 'cat_11', 'cat_12','cat_25', 'cat_26', 'cat_27', 'cat_29' ,'cat_30', 'cat_31' ,'cat_32',
                'cat_43', 'cat_44','cat_45' ,'cat_46','cat_47', 'cat_48',
           'cat_49', 'cat_50', 'cat_51', 'cat_52', 'cat_53', 'cat_54',
           'cat_55', 'cat_56', 'cat_57', 'cat_58','cat_59','cat_60', 'cat_61']


train_foo = train_foo[allcols]
test_foo = test_foo[allcols]
ttdata = ttdata[allcols]



#building Keras model with entity embedding
a = {}  #dict to store all embeddings and loop through to generate
models = []

for x in col2scale:
    a[x] = Sequential()
    a[x].add(Dense(1, input_dim=1))
    models.append(a[x])


for x in col2cat:
    count = ttdata[x].unique().shape[0]
    if count <= 10:
        a[x] = Sequential()
        a[x].add(Embedding(count+1, count-1, input_length=1))
        a[x].add(Reshape(target_shape=(count-1,)))
        models.append(a[x])
    if count >10 and count <=100:
        a[x] = Sequential()
        a[x].add(Embedding(count+1, 10, input_length=1))
        a[x].add(Reshape(target_shape=(10,)))
        models.append(a[x])
    if count >100 and count <=1000:
        a[x] = Sequential()
        a[x].add(Embedding(count+1, 25, input_length=1))
        a[x].add(Reshape(target_shape=(25,)))
        models.append(a[x])
    if count >1000:
        a[x] = Sequential()
        a[x].add(Embedding(count+1, 50, input_length=1))
        a[x].add(Reshape(target_shape=(50,)))
        models.append(a[x])   




#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
X_train, X_test, y_train, y_test = train_test_split(train_foo, labels, test_size=0.1, random_state=44, stratify = labels)
    
    
le_train = LabelEncoder()
y_train = le_train.fit_transform(y_train)
y_train = to_categorical(y_train)
    
le_test = LabelEncoder()
y_test = le_test.fit_transform(y_test)
y_test = to_categorical(y_test)



X_train = X_train.values
X_train = np.array(X_train)
X_train_list = []
i=0
while i < 306:
    foo_list = X_train[...,[i]]
    X_train_list.append(foo_list)
    i = i+1
X_train = X_train_list
    
X_test = X_test.values
X_test = np.array(X_test)
X_test_list = []
i=0
while i < 306:
    foo_list = X_test[...,[i]]
    X_test_list.append(foo_list)
    i = i+1
X_test = X_test_list



model = Sequential()
model.add(Merge(models, mode='concat'))
model.add(Dropout(0.2))
model.add(Dense(2000, init='normal')) #try init = uniform
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(500, init='normal'))
model.add(Activation('relu'))
model.add(Dense(21, init='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
ReduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, 
                                             verbose=2, mode='min',cooldown=5, min_lr=0.00001)

model.fit(X_train, y_train, validation_data = (X_test, y_test), 
          batch_size=128, nb_epoch=100,verbose = 2,callbacks=[earlyStopping,ReduceLR])