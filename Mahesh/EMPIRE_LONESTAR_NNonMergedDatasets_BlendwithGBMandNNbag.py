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
train_foo = pd.read_pickle("feat3_train_nn.pkl")
test_foo = pd.read_pickle("feat3_test_nn.pkl")
labels = pd.read_pickle("labels_foo.pkl")

train_foo['imparmt_pct'] = np.log(1+train_foo['imparmt_pct'])
test_foo['imparmt_pct'] = np.log(1+test_foo['imparmt_pct'])
train_foo.drop('econ_unemployment_py', inplace = True, axis = 1)
test_foo.drop('econ_unemployment_py', inplace = True, axis = 1)

train_aj = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test_aj = pd.read_csv('test_recode_12.csv.gz', compression="gzip")
#labels = train_foo['target'].copy()
train_aj.drop(['target', 'claim_id', 'loss'], inplace = True, axis = 1)
test_aj.drop(['claim_id'], inplace = True, axis = 1)

print("Done with datasets...")


cat2keep = ['cat_25', 'cat_47', 'cat_11', 'cat_48', 'cat_27', 'cat_26', 'cat_51', 'cat_5', 'cat_29', 'cat_45',
           'cat_50','cat_59','cat_28','cat_55','cat_30','cat_60','cat_15','cat_37','cat_8',
            'cat_43','cat_31','cat_53','cat_44','cat_52','cat_61','cat_57','cat_9','cat_58','cat_14',
            'cat_49','cat_56','cat_46']

cat_cols = [f for f in train_aj.columns if 'cat' in f]

cat_cols_drop = list(set(cat_cols) - set(cat2keep))

train_aj.drop(cat_cols_drop, inplace = True, axis = 1)
test_aj.drop(cat_cols_drop, inplace = True, axis = 1)

train_foo = pd.concat([train_foo.reset_index(drop = True), train_aj], axis=1)
test_foo = pd.concat([test_foo.reset_index(drop = True), test_aj], axis = 1)


train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

cat2counts = ['cat_25', 'cat_47', 'cat_11', 'cat_48', 'cat_27', 'cat_26', 'cat_51', 'cat_29', 'cat_45',
           'cat_50','cat_59','cat_55','cat_30','cat_60','cat_15',
            'cat_43','cat_31','cat_53','cat_44','cat_52','cat_61','cat_57','cat_9','cat_58','cat_14',
            'cat_49','cat_56','cat_46']
    
for x in cat2counts:
    #claims[x] = claims[x].astype(str)
    foo= ttdata[x].value_counts()
    ttdata[x] = np.log(ttdata[x].apply(lambda x: foo[x]))

print ("Done with cat 2 value counts")


train_foo['train_flag'] = 1
test_foo['train_flag'] = 0
ttdata = train_foo.append(test_foo)

col2scale = ['econ_unemployment_py',  'aww1','diagnosis_icd9_cd',
        'imparmt_pct', 'occ_desc1',  'diff_repo_loss', 'diff_impmt_loss',
       'diff_mmi_loss', 'diff_suit_loss', 'diff_paid_repo',
       'diff_paid_impmt', 'diff_paid_mmi', 'diff_paid_death',
       'diff_paid_return', 'diff_paid_suit', 'age_claimaint',
       'target_prev', 'adj_prev',  'target_median3', 
       'i_prev', 'm_prev', 'tpd_amt', 'l_prev', 'i_prev_avg3',
       'm_prev_avg3', 'econ_unemployment_py', 'paid_year', 'econ_gdp_py',
       'econ_gdp_ar_py', 'econ_price_py', 'econ_price_allitems_py',
       'econ_price_healthcare_py', 'econ_10yr_note_py', 'paid_i', 'paid_m',
       'paid_l', 'paid_o', 'pct_paid_i', 'pct_paid_m', 'pct_paid_im',
       'sign_paid_i', 'sign_paid_m', 'sign_paid_im', 'pct_paid_i_scale',
       'pct_paid_m_scale', 'pct_paid_im_scale', 'pct_paid_i_scale_sq',
       'pct_paid_m_scale_sq', 'pct_paid_im_scale_sq',
       'pct_paid_i_scale_dollar', 'pct_paid_m_scale_dollar',
       'pct_paid_im_scale_dollar', 'pct_paid_i_scale_sq_dollar',
       'pct_paid_m_scale_sq_dollar', 'pct_paid_im_scale_sq_dollar',
       'any_i', 'any_m', 'any_l', 'any_o', 'mean_i', 'mean_m', 'mean_l',
       'mean_o', 'mean_im', 'mean_all', 'sd_i', 'sd_m', 'sd_l', 'sd_o',
       'sd_im', 'sd_all', 'min_i', 'min_m', 'min_l', 'min_o', 'min_im',
       'min_all', 'max_i', 'max_m', 'max_l', 'max_o', 'max_im', 'max_all',
       'med_i', 'med_m', 'med_l', 'med_o', 'med_im', 'med_all', 'n_hist',
       'pct_paid_i_ave', 'pct_paid_m_ave', 'pct_paid_im_ave',
       'pct_paid_i_ave_scale', 'pct_paid_m_ave_scale',
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
            'cat_25', 'cat_47', 'cat_11', 'cat_48', 'cat_27', 'cat_26', 'cat_51', 'cat_29', 'cat_45',
           'cat_50','cat_59','cat_55','cat_30','cat_60','cat_15',
            'cat_43','cat_31','cat_53','cat_44','cat_52','cat_61','cat_57','cat_9','cat_58','cat_14',
            'cat_49','cat_56','cat_46']


for x in col2scale:
    #ttdata[x] = scale(ttdata[x]) #scale
    ms = MinMaxScaler(feature_range=(-1, 1))           #minmax
    ttdata[x] = ms.fit_transform(ttdata[x])
    
    
#Try with and without diagnosiscode dummies
col2dummy= ['slice', 'cas_aia_cds_1_2', 'clm_aia_cds_1_2', 'clmnt_gender',
       'clm_aia_cds_3_4', 'initl_trtmt_cd',
       'catas_or_jntcvg_cd', 'state', 'suit_matter_type',
       'law_limit_tt', 'law_limit_pt', 'law_cola', 'law_offsets',
       'law_ib_scheduled','cutoffyr_flag', 'lp1', 
            'cat_5', 'cat_28', 'cat_37', 'cat_8']

ttdata.drop(['target_prev', 'target_median3'], axis = 1, inplace = True)
ttdata = pd.get_dummies(ttdata, columns=col2dummy)

train_foo = ttdata[ttdata['train_flag']==1]
test_foo = ttdata[ttdata['train_flag']==0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)


if 1: #CV 0.2367 LB 0.23912

    if 1:
        
        #print ("Fold ->", i+1)
        X_train, X_test, y_train, y_test = train_test_split(train_foo, labels, test_size = 0.05, 
                                                            random_state= 44, stratify = labels)
    
    
        le_train = LabelEncoder()
        y_train = le_train.fit_transform(y_train)
        y_train = to_categorical(y_train)
    
        le_test = LabelEncoder()
        y_test = le_test.fit_transform(y_test)
        y_test = to_categorical(y_test)
    
        model = Sequential()
        model.add(Dense(500, input_dim = train_foo.shape[1] , init='normal')) #try init = uniform  510
        model.add(Activation('relu')) #600 0.2, 600 0.2, 600 0.2   0.2364
        model.add(Dropout(0.2)) #500 0.2, 500 0.2, 500 0.2  0.2362
        model.add(Dense(500, init='normal'))
        model.add(Activation('relu'))     
        model.add(Dropout(0.2))
        model.add(Dense(500, init='normal'))
        model.add(Activation('relu'))     
        model.add(Dropout(0.2))
        #model.add(Dense(500, init='normal'))
        #model.add(Activation('relu'))     
        #model.add(Dropout(0.2))
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
        #scores.append(score)
        pred = model.predict_proba(test_foo.values)
        #dataset_blend_train_rf2.iloc[test_index, 0:21] = y_val

preds = pd.DataFrame(pred)



#BLENDING New NN on merged Datasets WITH GBM11 from AJAY, NNBAG from MAHESH
gbm = pd.read_csv("gbm_m4_data11.csv.gz", compression = 'gzip')
nn1 = pd.read_pickle("STACK_LVL2_NNadam2_METAONLY_BAG5_400_TEST_CV_0.225847965856_.pkl")
nn2 = pd.read_pickle("STACK_LVL2_NNadam_METAONLY_BAG5_400_TEST_CV_0.225769130374_.pkl")
nn3 = pd.read_pickle("STACK_LVL2_NNadam_METAONLY_BAG5_400_TEST_CV_0.225798501273_.pkl")
nn4 = pd.read_pickle("STACK_LVL2_NNadam3_METAONLY_BAG5_400_TEST_CV_0.225823880855_.pkl")
nn5 = pd.read_pickle("STACK_LVL2_NNadam3_METAONLY_BAG10_01D_400_TEST_CV_0.225896988973_.pkl")
nn6 = pd.read_pickle("STACK_LVL2_NNadam_METAONLY_BAG10_01D_400_TEST_CV_0.225847383425_.pkl")

nn_bag = nn1.copy()
for x in nn_bag.columns.values:
    nn_bag[x] = (((nn1[x] + nn2[x])/2)  +  ((nn3[x] + nn4[x])/2)  + nn5[x] + nn6[x])/4

    
gbm.drop('claim_id', inplace = True, axis = 1)


gbm = pd.DataFrame(gbm.values)


blend = preds.copy()

for x in blend.columns.values:
    blend[x] = (preds[x] + gbm[x] + nn_bag[x])/3

blend.to_csv("blend_gbm_nnbag_nnwithcats-1-28.csv")